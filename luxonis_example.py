import blobconverter
import cv2
import numpy as np
from pyzbar import pyzbar

from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import TextPosition


def callback(packet):
    for detection in packet.detections:
        bbox = detection.top_left[0], detection.top_left[1], detection.bottom_right[0], detection.bottom_right[1]
        # Expand bounding box
        bbox = (max(0, bbox[0] * 0.9), max(0, bbox[1] * 0.9),
                min(packet.frame.shape[1], bbox[2] * 1.1), min(packet.frame.shape[0], bbox[3] * 1.1))
        bbox = np.intp(bbox)
        cropped_qr = packet.frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]  # crop QR code
        cropped_qr = 255 - cropped_qr  # invert colors for revert black and white

        # Decode QR code
        arcodes = pyzbar.decode(cropped_qr)
        for barcode in barcodes:
            barcode_info = barcode.data.decode('utf-8')
            # Add text to the frame
            packet.visualizer.add_text(barcode_info, size=1, bbox=bbox, position=TextPosition.MID, outline=True)

    frame = packet.visualizer.draw(packet.frame)
    cv2.imshow('QR code recognition', frame)


with OakCamera() as oak:
    color = oak.create_camera('color', fps=30)
    nn_path = blobconverter.from_zoo(name="qr_code_detection_384x384", zoo_type="depthai")
    nn = oak.create_nn(nn_path, color, nn_type='mobilenet')
    nn.config_nn(resize_mode='stretch')

    visualizer = oak.visualize(nn, record_path='qr_video.mp4', callback=callback)
    oak.start(blocking=True)