# Opticalib

This tool is designed for multi-camera calibration tasks, and estimates intrinsics and extrinsics of cameras. It also provides a blender scene to visualizate the results. It natively supports data collection using Basler GigE cameras.

## Installation

1. This project uses the uv python package manager: https://docs.astral.sh/uv/

    You can install uv on Linux with:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    
    If you want to install uv in a different way, check the documentation on https://docs.astral.sh/uv/getting-started/installation/

2. Clone the repository. Be sure to put the --recursive option to properly install also git submodules:
    
    ```bash
    git clone --recursive https://github.com/emcaruso/opticalib.git
    ```
    Navigate to the project directory and install the required dependencies on the virtual environment with uv:
    ```bash
    cd opticalib
    uv sync
    ```

3. To acquire images with Basler cameras, ensure that cameras are properly configured following the Basler documentation: https://docs.Baslerweb.com/cameras. Be sure that the network and ip addresses are properly configured. You can use PylonViewer to check if cameras work: https://docs.Baslerweb.com/overview-of-the-pylon-viewer.

## Usage


### Print boards

This calibration tool currently works with charuco boards only.
in `configs/charuco_boards/` there is a directory where you can put different config files relative to charuco boards. The format is the following:

```yaml
type: charuco                  # ❌ Do not change

boards:
  number_x_square: 11          # number of squares in the X direction
  number_y_square: 9           # number of squares the Y direction
  resolution_x: 2200           # horizontal resolution in pixel
  resolution_y: 1800           # vertical resolution in pixel
  length_square: 0.024         # parameters on the marker (can be kept as it is)
  length_marker: 0.018         # parameters on the marker (can be kept as it is)
  n_boards: 1                  # number of boards used for calibration (for overlapping camera 1 is enough ...)
  square_size: 0.0245          # size of each square of the board in meters
  aruco_n: 6                   # size of aruco marker

detector:                      # aruco detection parameters
  adaptiveThreshWinSizeStep: 5
```

As a first step, you have to generate charuco images. To generate charuco images, run the script `scripts/get_charuco_images.sh`. A window will appear, and you can choose a .yaml file in the `charuco_boards` folder. After choosing the file, charuco images are saved in `results/charuco_boards`, and will be displayed on the screen.
Then you have to print charuco boards.
After printing the boards, you have also to correct the `square_size` value in the .yaml file if it's different.
when using multiple boards, you should put them on a rigid structure as their relative position and orientation must not change when acquiring images.


### Calibration

The calibration process can be run in 3 different modalities

- Intrinsics calibration
- Intrinsics calibration + global calibration
- Intrinsics calibration + global calibration while keeping fixed intrinsics


