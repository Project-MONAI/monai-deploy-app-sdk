from pathlib import Path

from skimage.filters import median
from skimage.io import imread, imsave


def process_image(input_path, output_path):
    """
    Processes an image and saves it to a new file.
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        input_path = next(input_path.glob("*.*"))  # take the first file

    data_in = imread(input_path)[:, :, :3]  # discard alpha channel if exists

    data_out = median(data_in)

    output_folder = Path(output_path)
    output_path = output_folder / "median.png"
    imsave(output_path, data_out)


if __name__ == "__main__":
    process_image("/input", "/output")
