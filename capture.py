import cv2
import cvui
import click
import numpy as np
from pathlib import Path
from functools import partial

from scripts.rgb_manager import RGBCaptureManager
from scripts.utils import (
    create_inference,
    get_time,
    make_save_dir,
    clean_save_dir,
    save_image,
    colorize_mask,
    convert_image_to_infer,
    infer_image,
    get_masked_image_with_segmentation,
    scaling_int,
    get_number_of_saved_image
)


SCRIPT_DIR = Path(__file__).parent.resolve()


@click.command()
@click.option("--toml-path", "-t", default=f"{SCRIPT_DIR}/cfg/camera_parameter.toml")
@click.option("--directory-for-save", "-s", default=f"{SCRIPT_DIR}/data")
@click.option("--config-name", "-c", default=f"{SCRIPT_DIR}/cfg/semantic_segmentation_multi_class.toml")
@click.option("--rgb_rate", "-r", default=0.4)
@click.option("--scale-for-visualization", "-sc", default=0.4)
def main(toml_path, directory_for_save, config_name, rgb_rate, scale_for_visualization):
    rgb_manager = RGBCaptureManager(toml_path)
    inference = create_inference(config_name)
    width, height = rgb_manager.size
    scaling = partial(scaling_int, scale=scale_for_visualization)

    width_resized = scaling(width)
    height_resized = scaling(height)
    frame = np.zeros((height_resized + 300, width_resized * 2, 3), np.uint8)

    WINDOW_NAME = "Capture"
    cvui.init(WINDOW_NAME)
    while True:
        frame[:] = (49, 52, 49)
        key = cv2.waitKey(10)
        status = rgb_manager.update()
        if not status:
            continue

        rgb_image_raw = rgb_manager.read()
        rgb_image_masked = get_masked_image_with_segmentation(rgb_image_raw, rgb_manager, inference, rgb_rate)

        number_of_saved_frame = get_number_of_saved_image(directory_for_save)
        cvui.printf(frame, 50, height_resized + 50, 0.8, 0x00FF00, "Number of Captured Images : %d", number_of_saved_frame)
        if cvui.button(frame, 50, height_resized + 110, 200, 100, "capture image") or key & 0xFF == ord("s"):
            save_image(rgb_image_raw, directory_for_save)

        if cvui.button(frame, 300, height_resized + 110, 200, 100, "erase"):
            clean_save_dir(directory_for_save)

        rgb_image_resized = cv2.resize(rgb_image_raw, (width_resized, height_resized))
        masked_image_resized = cv2.resize(rgb_image_masked, (width_resized, height_resized))
        frame[0:height_resized, 0:width_resized, :] = rgb_image_resized
        frame[0:height_resized, width_resized:(width_resized * 2), :] = masked_image_resized

        if key == 27 or key == ord("q"):
            break

        cvui.update()
        cvui.imshow(WINDOW_NAME, frame)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()