import depthai
from typing import Callable, Optional, Generator
from contextlib import contextmanager
import numpy as np

class Rect():
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def get_tuple(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

class Camera():
    def __init__(self,
                 mxid: Optional[str] = None,
                 sensor_resolution: depthai.ColorCameraProperties.SensorResolution = depthai.ColorCameraProperties.SensorResolution.THE_4_K,
                 crop_rect: Rect = Rect(.0, .0, 1.0, 1.0),
                 frame_callback: Optional[Callable] = None,
                 ) -> None:
        self._sensor_resolution = sensor_resolution
        self.crop_rect = crop_rect
        self.frame_callback = self._frame_callback
        if frame_callback is not None:
            self.frame_callback = frame_callback
        self._mxid = mxid
        
        self._control_stream = "control"
        self._config_stream = "config"
        self._manip_stream = "manip"
        self._video_stream = "video"

        self._pipeline = self._get_pipeline()

        self._manip_config = depthai.ImageManipConfig()
        self._manip_config.setFrameType(depthai.RawImgFrame.Type.BGR888p)
        self._manip_config.setCropRect(self.crop_rect.get_tuple())
        self._manip_config.setKeepAspectRatio(False)

        self.exposure = 20_000
        self.iso = 800
        self.focus = 140
        self.contrast = 0
        self._camera_control = depthai.CameraControl()
        self._camera_control.setManualExposure(self.exposure, self.iso)

        self.frame = np.full(shape=self.max_shape, fill_value=255, dtype=np.uint8)


    def _get_pipeline(self) -> depthai.Pipeline:
        pipeline = depthai.Pipeline()

        self._camera = pipeline.create(depthai.node.ColorCamera)
        self._camera.setResolution(self._sensor_resolution)
        self.max_shape = (self._camera.getResolutionHeight(), self._camera.getResolutionWidth(), 3)

        x_control_in = pipeline.create(depthai.node.XLinkIn)
        x_control_in.setStreamName(self._control_stream)
        x_control_in.out.link(self._camera.inputControl)

        x_config_in = pipeline.create(depthai.node.XLinkIn)
        x_config_in.setStreamName(self._config_stream)
        x_config_in.out.link(self._camera.inputConfig)

        manip = pipeline.create(depthai.node.ImageManip)
        manip.initialConfig.setFrameType(depthai.RawImgFrame.Type.BGR888p)
        manip.setMaxOutputFrameSize(self._camera.getResolutionHeight()*self._camera.getResolutionWidth()*3)
        self._camera.video.link(manip.inputImage)

        x_manip_in = pipeline.create(depthai.node.XLinkIn)
        x_manip_in.setStreamName(self._manip_stream)
        x_manip_in.out.link(manip.inputConfig)

        x_video_out = pipeline.create(depthai.node.XLinkOut)
        x_video_out.setStreamName(self._video_stream)
        x_video_out.input.setBlocking(False)
        x_video_out.input.setQueueSize(1)
        manip.out.link(x_video_out.input)

        return pipeline
    
    def _frame_callback_wrapper(self, frame: depthai.RawImgFrame) -> None:
        self.frame = frame.getCvFrame()
        self.frame_callback(self.frame)

    # not used if frame_callback is specified
    def _frame_callback(self, frame: np.ndarray) -> None:
        return

    def _setup_queues(self) -> None:
        if self.frame_callback is not None:
            self._device.getOutputQueue(name=self._video_stream, maxSize=1, blocking=False).addCallback(self._frame_callback_wrapper)
        self._control_queue = self._device.getInputQueue(name=self._control_stream, maxSize=1, blocking=False)
        self._config_queue = self._device.getInputQueue(name=self._config_stream, maxSize=1, blocking=False)
        self._manip_queue = self._device.getInputQueue(name=self._manip_stream, maxSize=1, blocking=False)

        self._control_queue.send(self._camera_control)
        self._manip_queue.send(self._manip_config)

    @contextmanager
    def connect(self) -> Generator[None, None, None]:
        if self._mxid is not None:
            devices = depthai.Device.getAllAvailableDevices()
            device_info = None
            for _device_info in devices:
                if _device_info.mxid == self._mxid:
                    device_info = _device_info
            if device_info is None:
                raise Exception(f"could not find camera with mxid {self._mxid}")
            with depthai.Device(self._pipeline, device_info, depthai.UsbSpeed.SUPER) as device:
                self._device = device
                self._setup_queues()
                try:
                    yield
                finally:
                    pass
        else:
            with depthai.Device(self._pipeline, depthai.UsbSpeed.SUPER) as device:
                self._device = device
                self._setup_queues()
                try:
                    yield
                finally:
                    pass

    def set_focus(self, focus: int) -> None:
        self.focus = focus
        self._camera_control.setManualFocus(self.focus)
        self._control_queue.send(self._camera_control)

    def set_exposure(self, exposure: int) -> None:
        self.exposure = exposure
        self._camera_control.setManualExposure(self.exposure, self.iso)
        self._control_queue.send(self._camera_control)

    def set_iso(self, iso: int) -> None:
        self.iso = iso
        self._camera_control.setManualExposure(self.exposure, self.iso)
        self._control_queue.send(self._camera_control)

    def set_contrast(self, contrast: int) -> None:
        self.contrast = contrast
        self._camera_control.setContrast(self.contrast)
        self._control_queue.send(self._camera_control)

    def set_crop_xmin(self, value: float) -> None:
        self.crop_rect.xmin = value
        self._manip_config.setCropRect(self.crop_rect.get_tuple())
        self._manip_queue.send(self._manip_config)

    def set_crop_ymin(self, value: float) -> None:
        self.crop_rect.ymin = value
        self._manip_config.setCropRect(self.crop_rect.get_tuple())
        self._manip_queue.send(self._manip_config)

    def set_crop_xmax(self, value: float) -> None:
        self.crop_rect.xmax = value
        self._manip_config.setCropRect(self.crop_rect.get_tuple())
        self._manip_queue.send(self._manip_config)

    def set_crop_ymax(self, value: float) -> None:
        self.crop_rect.ymax = value
        self._manip_config.setCropRect(self.crop_rect.get_tuple())
        self._manip_queue.send(self._manip_config)

    def get_config(self) -> dict:
        res = {
            "xmin": self.crop_rect.xmin,
            "ymin": self.crop_rect.ymin,
            "xmax": self.crop_rect.xmax,
            "ymax": self.crop_rect.ymax,
            "focus": self.focus,
            "exposure": self.exposure,
            "iso": self.iso,
            "contrast": self.contrast,
        }
        return res