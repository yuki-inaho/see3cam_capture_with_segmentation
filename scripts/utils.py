import os
import toml
import shutil
from pytz import timezone
from datetime import datetime
from pathlib import Path
from scripts.inference_manager import InferSegmentation
from scripts.rgb_manager import RGBCaptureManager
import numpy as np
import cv2

BASE_DIR_PATH = Path(__file__).resolve().parents[1]
dict_idx2color = {
    0: (0, 0, 0),
    1: (110, 110, 255),  # Oyagi
    2: (150, 249, 152),  # Aspara
    3: (255, 217, 81),  # Ground
    4: (252, 51, 255),  # Tube
    5: (84, 223, 255),  # Pole
}


def create_inference(config_path=f"{BASE_DIR_PATH}/cfg/semantic_segmentation.toml"):
    if len(config_path) == 0:
        raise "no config file given"
    print(config_path)
    with open(str(config_path), "r") as f:
        config = toml.load(f)

    return InferSegmentation(
        weights=str(config["weights"]),
        architecture=config["architecture"],
        encoder=config["encoder"],
        depth=config["depth"],
        in_channels=config["in_channels"],
        classes=config["classes"],
        activation=config["activation"],
        resize=config["resize"],
        gpu=config["gpu_id"],
    )


def add_dummy_dim(image):
    return image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))


def convert_img_dim(image):
    return image.reshape(image.shape[1], image.shape[2], image.shape[0])


def get_overlay_rgb_image(rgb_image, mask, rgb_rate=0.6, mask_rate=0.4):
    if len(mask.shape) > 2:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        segmentation_overlay_rgb = cv2.addWeighted(rgb_image, rgb_rate, mask, mask_rate, 2.5)
    else:
        mask_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        nonzero_idx = np.where(mask > 0)
        mask_image[nonzero_idx[0], nonzero_idx[1], :] = (0, 0, 255)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        segmentation_overlay_rgb = cv2.addWeighted(rgb_image, rgb_rate, mask_image, mask_rate, 2.5)
    return segmentation_overlay_rgb


def get_time():
    utc_now = datetime.now(timezone("UTC"))
    jst_now = utc_now.astimezone(timezone("Asia/Tokyo"))
    time = str(jst_now).split(".")[0].split(" ")[0] + "_" + str(jst_now).split(".")[0].split(" ")[1]
    return time


def make_save_dir(save_dir_path):
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)


def clean_save_dir(save_dir_path):
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path)
    os.mkdir(save_dir_path)


def save_image(see3cam_rgb_img, save_dir):
    time = get_time()
    cv2.imwrite(os.path.join(save_dir, "{}.png".format(time)), see3cam_rgb_img)
    cv2.waitKey(10)


def colorize_mask(mask_image, n_label):
    mask_colorized = np.zeros([mask_image.shape[0], mask_image.shape[1], 3], dtype=np.uint8)
    for l in range(n_label + 1):
        mask_indices_lth_label = np.where(mask_image == l)
        mask_colorized[mask_indices_lth_label[0], mask_indices_lth_label[1], :] = dict_idx2color[l]
    return mask_colorized


def convert_image_to_infer(image_bgr, rgb_manager):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image


def infer_image(image, inference):
    mask_image_tmp = convert_img_dim(inference(add_dummy_dim(image)))
    mask_image = colorize_mask(mask_image_tmp, 5)
    return mask_image


def get_masked_image_with_segmentation(rgb_image_raw, rgb_manager: RGBCaptureManager, inference: InferSegmentation, rgb_rate):
    rgb_image = convert_image_to_infer(rgb_image_raw, rgb_manager)
    segmentation_mask = infer_image(rgb_image, inference)
    rgb_image_masked = get_overlay_rgb_image(rgb_image, segmentation_mask, rgb_rate=rgb_rate, mask_rate=1 - rgb_rate)
    return rgb_image_masked


def scaling_int(int_num, scale):
    return int(int_num * scale)


def get_number_of_saved_image(directory_for_save):
    number_of_saved_frame = len([str(path) for path in Path(directory_for_save).glob("*.png")])
    return number_of_saved_frame
