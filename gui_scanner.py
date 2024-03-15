import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from pyzbar.pyzbar import decode, ZBarSymbol
import depthai
import os
from contextlib import ExitStack
from functools import partial
import trio
import trio_typing
import trio_parallel
from multiprocessing import Queue, Manager
from typing import Tuple
from queue import Full as QueueFullException
from queue import Empty as QueueEmptyException

from camera import Camera, Rect

frame_size = (432, 768)

def resize_and_pad(img, size, padColor=(0, 0, 0, 255)):   
    h, w = img.shape[:2]
    sh, sw = size
    
    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    
    else: # stretching image
        interp = cv2.INTER_CUBIC
    
    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh
    
    if (saspect >= aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    
    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    
    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    
    return scaled_img

def convert_frame_cv2_to_dpg(frame: np.ndarray) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    frame = resize_and_pad(frame, frame_size)
    frame = np.ravel(frame)
    frame = np.asfarray(frame, dtype='float')
    frame = np.true_divide(frame, 255.0)
    return frame

async def qr_worker_handler(task_status: trio_typing.TaskStatus[Tuple[Queue, Queue]]) -> None:
    with Manager() as manager:
        input_frame_queue = manager.Queue(maxsize=2)
        result_output_queue = manager.Queue(maxsize=2)
        task_status.started((input_frame_queue, result_output_queue))
        await trio_parallel.run_sync(qr_worker, input_frame_queue, result_output_queue, cancellable=True)

def qr_worker(frame_queue: Queue, result_queue: Queue) -> None:
    while True:
        frame = frame_queue.get()
        result = decode(frame, symbols=[ZBarSymbol.QRCODE, ZBarSymbol.CODE128])
        try:
            result_queue.put_nowait(result)
        except QueueFullException:
            print("vafan")

async def main() -> None:
    with ExitStack() as exit_stack:
        dpg.create_context()

        cameras = {}
        camera_side = Camera(mxid="19443010E167281300")
        try:
            exit_stack.enter_context(camera_side.connect())
            cameras["side"] = camera_side
        except Exception as err:
            print(err)
        camera_top = Camera(mxid="19443010D13B291300", crop_rect=Rect(0.10199999809265137, 0.11999999731779099, 0.8790000081062317, 0.5740000009536743))
        try:
            exit_stack.enter_context(camera_top.connect())
            cameras["top"] = camera_top
        except Exception as err:
            print(err)

        camera_frames = {}
        annotation_frames = {}

        def frame_callback(camera_name: str, frame: np.ndarray) -> None:
            camera_frames[camera_name] = frame

        with dpg.texture_registry(show=False):
            for camera_name, camera in cameras.items():
                camera_frames[camera_name] = np.full(camera.max_shape, 0, dtype="uint8")
                annotation_frames[camera_name] = np.full(camera.max_shape, 0, dtype="uint8")
                dpg.add_dynamic_texture(width=frame_size[1], height=frame_size[0], default_value=convert_frame_cv2_to_dpg(camera_frames[camera_name]), tag=f"texture_{camera_name}")
                camera.frame_callback = partial(frame_callback, camera_name)

        def focus_callback(sender, app_data, user_data):
            value = dpg.get_value(sender)
            user_data.set_focus(value)

        def exposure_callback(sender, app_data, user_data):
            value = dpg.get_value(sender)
            user_data.set_exposure(value)

        def iso_callback(sender, app_data, user_data):
            value = dpg.get_value(sender)
            user_data.set_iso(value)

        def contrast_callback(sender, app_data, user_data):
            value = dpg.get_value(sender)
            user_data.set_contrast(value)

        def xmin_callback(sender, app_data, user_data):
            value = dpg.get_value(sender)
            user_data.set_crop_xmin(value)

        def ymin_callback(sender, app_data, user_data):
            value = dpg.get_value(sender)
            user_data.set_crop_ymin(value)

        def xmax_callback(sender, app_data, user_data):
            value = dpg.get_value(sender)
            user_data.set_crop_xmax(value)

        def ymax_callback(sender, app_data, user_data):
            value = dpg.get_value(sender)
            user_data.set_crop_ymax(value)

        def snap_callback(sender, app_data, user_data):
            i = 0
            while os.path.exists(f"output/{i}.png"):
                i += 1
            cv2.imwrite(f"output/{i}.png", user_data.frame)

        def print_config_callback(sender, app_data, user_data):
            print(user_data.get_config())

        with dpg.window() as window:
            with dpg.table():
                for camera_name in cameras:
                    dpg.add_table_column(label=camera_name)
                with dpg.table_row():
                    for camera_name, camera in cameras.items():
                        with dpg.group():
                            dpg.add_slider_float(label="xmin", min_value=0, max_value=1, width=frame_size[1], callback=xmin_callback, user_data=camera)
                            dpg.add_slider_float(label="xmax", min_value=0, max_value=1, width=frame_size[1], default_value=1, callback=xmax_callback, user_data=camera)
                            with dpg.group(horizontal=True):
                                dpg.add_image(texture_tag=f"texture_{camera_name}")
                                dpg.add_slider_float(label="ymin", vertical=True, min_value=1, max_value=0, height=frame_size[0], callback=ymin_callback, user_data=camera)
                                dpg.add_slider_float(label="ymax", vertical=True, min_value=1, max_value=0, height=frame_size[0], default_value=1, callback=ymax_callback, user_data=camera)
                            dpg.add_button(label="snap!", callback=snap_callback, user_data=camera)
                            dpg.add_text("settings:")
                            dpg.add_slider_int(label="focus", callback=focus_callback, user_data=camera, default_value=140, min_value=1, max_value=255)
                            dpg.add_slider_int(label="exposure", callback=exposure_callback, user_data=camera, default_value=camera.exposure, min_value=1, max_value=33_000)
                            dpg.add_slider_int(label="iso", callback=iso_callback, user_data=camera, default_value=camera.iso, min_value=100, max_value=1_600)
                            dpg.add_slider_int(label="contrast", callback=contrast_callback, user_data=camera, default_value=0, min_value=-10, max_value=10)
                            dpg.add_button(label="print config", callback=print_config_callback, user_data=camera)
                            #dpg.add_text(tag=f"xmin_text_{camera_name}")
                            #dpg.add_text(tag=f"ymin_text_{camera_name}")
                            #dpg.add_text(tag=f"xmax_text_{camera_name}")
                            #dpg.add_text(tag=f"ymax_text_{camera_name}")
                            #dpg.add_text(tag=f"focus_text_{camera_name}")
                            #dpg.add_text(tag=f"exposure_text_{camera_name}")
                            #dpg.add_text(tag=f"iso_text_{camera_name}")
                            #dpg.add_text(tag=f"contrast_text_{camera_name}")
                

        dpg.set_primary_window(window, True)
        dpg.create_viewport(title="scanner")
        dpg.setup_dearpygui()
        dpg.show_viewport()

        async with trio.open_nursery() as nursery:
            image_queues = {}
            result_queues = {}
            for camera_name in cameras:
                image_queues[camera_name], result_queues[camera_name] = await nursery.start(qr_worker_handler)

            while dpg.is_dearpygui_running():
                for camera_name, camera in cameras.items():
                    
                    try:
                        image_queues[camera_name].put_nowait(camera.frame)
                    except QueueFullException:
                        pass

                    decoded = None
                    try:
                        decoded = result_queues[camera_name].get_nowait()
                    except QueueEmptyException:
                        pass
                    
                    camera_frame = camera_frames[camera_name]
                    if annotation_frames[camera_name].shape != camera_frame.shape or decoded:
                        annotation_frames[camera_name] = camera_frame.copy()

                    if decoded:
                        print(camera_name, end=": ")
                        for d in decoded:
                            s = d.data.decode()
                            print(f"{s} ", end="")
                            annotation_frames[camera_name] = cv2.rectangle(annotation_frames[camera_name], (d.rect.left, d.rect.top), (d.rect.left + d.rect.width, d.rect.top + d.rect.height), (0, 0, 255), 5)
                            annotation_frames[camera_name] = cv2.putText(annotation_frames[camera_name], s, (d.rect.left, d.rect.top + d.rect.height), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
                        print()

                    try:
                        combined = cv2.addWeighted(annotation_frames[camera_name], 0.5, camera_frame, 0.5, 0)
                    except Exception:
                        print(annotation_frames[camera_name].shape)
                        print(camera_frame.shape)
                        raise
                    combined = convert_frame_cv2_to_dpg(combined)
                    dpg.set_value(f"texture_{camera_name}", combined)

                    #dpg.set_value(f"xmin_text_{camera_name}", f"xmin: {camera.crop_rect.xmin}")
                    #dpg.set_value(f"ymin_text_{camera_name}", f"ymin: {camera.crop_rect.ymin}")
                    #dpg.set_value(f"xmax_text_{camera_name}", f"xmax: {camera.crop_rect.xmax}")
                    #dpg.set_value(f"ymax_text_{camera_name}", f"ymax: {camera.crop_rect.ymax}")
                    #dpg.set_value(f"focus_text_{camera_name}", f"focus: {camera.focus}")
                    #dpg.set_value(f"exposure_text_{camera_name}", f"exposure: {camera.exposure}")
                    #dpg.set_value(f"iso_text_{camera_name}", f"iso: {camera.iso}")
                    #dpg.set_value(f"contrast_text_{camera_name}", f"contrast: {camera.contrast}")

                dpg.render_dearpygui_frame()
                await trio.sleep(0)

            nursery.cancel_scope.cancel()

    dpg.destroy_context()

if __name__ == "__main__":
    trio.run(main)