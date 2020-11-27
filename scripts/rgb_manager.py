import cv2
import toml
import numpy as np
from scripts.camera_parameter import IntrinsicParameter


class LensUndistorter:
    def __init__(self, K_rgb, distortion_params, image_width, image_height, enable_tps=False):
        self.distortion_params = distortion_params
        self.DIM = (image_width, image_height)
        if enable_tps:
            self.K_rgb = K_rgb
            _map1, _map2 = cv2.fisheye.initUndistortRectifyMap(self.K_rgb, self.distortion_params, np.eye(3), self.K_rgb, self.DIM, cv2.CV_16SC2)
        else:
            self.K_rgb_raw = K_rgb
            self.K_rgb = cv2.getOptimalNewCameraMatrix(self.K_rgb_raw, self.distortion_params, self.DIM, 0)[0]
            _map1, _map2 = cv2.fisheye.initUndistortRectifyMap(self.K_rgb_raw, self.distortion_params, np.eye(3), self.K_rgb, self.DIM, cv2.CV_16SC2)
        self.map1 = _map1
        self.map2 = _map2
        self.P_rgb = (self.K_rgb[0][0], 0.0, self.K_rgb[0][2], 0.0, 0.0, self.K_rgb[1][1], self.K_rgb[1][2], 0.0, 0.0, 0.0, 1.0, 0.0)

    def correction(self, image):
        return cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def correction_with_mask(self, mask):
        return cv2.remap(mask, self.map1, self.map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    @property
    def K(self):
        return self.K_rgb

    @property
    def P(self):
        return self.P_rgb


class RGBCaptureManager:
    def __init__(self, toml_path, enable_undistortion=True):
        self._setting(toml_path)
        """
        Read toml setting file and set
            self.device_id, self.fps
            self.image_{width, height}
            self.image_{width_raw, height_raw}
            self.intrinsic_params, self.intrinsic_params_raw
            self.K_rgb, self.K_rgb_raw,
        """
        self.stopped = False
        self.is_grabbed = False
        self.frame = None

        # Set Video Capture Module
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        self.cap.set(cv2.CAP_PROP_FPS, int(self.fps))

        self.enable_undistortion = enable_undistortion
        if self.enable_undistortion:
            # Enable barrel distortion correction, and Disable TPS undistortion case
            self.lens_undistorter = LensUndistorter(self.K_rgb, self.distortion_params, self.image_width, self.image_height, enable_tps=False)
        self.P_rgb = np.c_[self.K_rgb, np.repeat(1.0, 3)]

    def _setting(self, toml_path):
        toml_dict = toml.load(open(toml_path))

        # Set Camera Settings
        self.device_id = toml_dict["Rgb"]["device_id"]
        self.image_width = toml_dict["Rgb"]["width"]
        self.image_height = toml_dict["Rgb"]["height"]
        self.fps = toml_dict["Rgb"]["fps"]

        # Set Camera Parameters
        intrinsic_elems = ["fx", "fy", "cx", "cy"]
        self.intrinsic_params = IntrinsicParameter()
        self.intrinsic_params.set_intrinsic_parameter(*[toml_dict["Rgb"][elem] for elem in intrinsic_elems])
        self.intrinsic_params.set_image_size(*[toml_dict["Rgb"][elem] for elem in ["width", "height"]])
        self.K_rgb = np.array(
            [[self.intrinsic_params.fx, 0, self.intrinsic_params.cx], [0, self.intrinsic_params.fy, self.intrinsic_params.cy], [0, 0, 1]]
        )

        self.distortion_params = np.array([toml_dict["Rgb"]["k{}".format(i + 1)] for i in range(4)])

    def update(self):
        # For latency related with buffer
        for i in range(3):
            status_rgb, rgb_image = self.cap.read()
        if not status_rgb:
            return False

        if self.enable_undistortion:
            self.frame = self.lens_undistorter.correction(rgb_image)
        else:
            self.frame = rgb_image
        return True

    def read(self):
        return self.frame

    def enable_tps_undistortion(self, distortion_info_mat):
        self.distortion_info_mat = distortion_info_mat
        self.tps_is_ready = True

    def get_param_matrix(self):
        return self.K_rgb, self.P_rgb

    def get_intrinsic_parameters(self):
        return self.intrinsic_params

    @property
    def size(self):
        return self.image_width, self.image_height

    @property
    def K(self):
        return self.K_rgb

    @property
    def K_raw(self):
        return self.K_rgb_raw
