# Opticalib

This tool is designed for multi-camera calibration tasks, and estimates intrinsics and extrinsics of cameras. in addition to providing the estimated parameters, it also generates data to visualize the estimated scene. It natively supports data collection using Basler GigE cameras.

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
Then you can to print charuco boards.
After printing the boards, you have also to correct the `square_size` value in the .yaml file if it's different.
If you use multiple boards (change the parameter `n_boards`), they will be considered as a single object, and their relative position and orientation must not change when acquiring images. In that case, you should put the charuco boards on a rigid structure


### Calibration

#### Explanation of the method

- Scene representation:
In this project, we use the perspective camera model used in OpenCV (https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).  Orientations are represented using Euler angles, using the YXZ convention. The 3D scene is composed by cameras, and an object with varying pose on time. The object consists on a set of charuco boards with fixed relative poses.

- Initialization:
The calibration method starts with an initialization step using OpenCV. It initializes the world reference frame and the relative rigid transformation between charuco boards when using multiple boards. Specifically, it initializes the world reference frame on the center of the first charuco board, selecting the images captured when as much points as possible are visible on all the images. We can also decide whether to keep the prior focal length specified in the config file while keeping the principal point at the center of the image, or to initialize it by calibrating with OpenCV.

- Precalibration: 
This step consists on solving a gradient-based, iterative, optimization problem with torch to optimize object poses while minimizing the reprojection error. We can also decide to constraint the object to lie on common world axes or to share some euler angles.

- Calibration:
Also the calibration step relies on an optimization problem like precalibration. This step, estimates the involved cameras parameters, such as the projection matrix and the distortion coefficients when performing intrinsics calibration, position and orientation of cameras when running extrinsics calibraiton, and both of them when running global calibration.

##### 1. Intrinsics calibration

Calibrate only the intrinsics parameters of cameras. You can run `collect_intrinsics.sh` to collect data, and run `calibrate_intrinsics` once data is collected. You can also run `collect_and_calibrate_intrinsics.sh` to do both steps. You can change parameters of intrinsics calibration looking at the config file `configs/cam_calib_intrinsics.yaml`:

##### 2. Extrinsics calibration

Calibrate only the extrinsics parameters of cameras, i.e. position and orientations. You can run `collect_extrinsics.sh` to collect data, and run `calibrate_extrinsics` once data is collected. You can also run `collect_and_calibrate_extrinsics.sh` to do both steps.

##### 3. Global calibration

Jointly calibrate intrinsics and extrinsics parameters of cameras. You can run `collect_global.sh` to collect data, and run `calibrate_global` once data is collected. You can also run `collect_and_calibrate_global.sh` to do both steps.

### Collection

When collecting images, the program will automatically detect connected Basler cameras, and a session will start to collect images. Parameters for Basler cameras can be set in the config file `configs/collector/cameras/basler.yaml`. You can set collection parameters such as

- `exposure_time`: the exposure time measured in microseconds
- `gamma`: the gamma value applied for gamma correction
- `gain`: the intensity gain applied to the image
- `fps`: refresh rate of cameras
- `pixel_format/val`: to choose between RGB and grayscale images
- `crop/do`: If true, images are cropped according to the ROI specified in the .pfs config file of basler cameras in `data/pfs_files`.
- `crop/slot`: The slot associated to the ROI in the .pfs config file

When working with cropped images, the method will automatically consider the offset of the principal point in the intrinsics parameters.

You can run `collect_intrinsics.sh` for intrinsics calibration, or `collect_extrinsics.sh` for extrinsic (or global) calibration.

When collecting images for intrinsics, you can collect a set of images for each camera at time. For intrinsics calibration, we suggest to capture at least 10 images covering a big portion of the image, especially the borders, with the charuco board. For cameras with high focal length, and long min depth of field, consider to use big charuco boards.

While collecting images for extrinsics or global calibration instead, cameras will grab images simultaneously.
There are mainly two modalities to capture images, and they can be set in the config files `configs/collector/collector_[global/intrinsics]`.

```yaml
defaults:
  - cameras: basler       # ❌ Do not change
  - lights: none          # ❌ Do not change
  - strategies: none      # ❌ Do not change
  - postprocessings: none # ❌ Do not change

paths:
  save_dir: "${paths.collection_dir}" # ❌ Do not change

mode:
  val: manual                                 # ✅ manual or automatic 
  valid_options:                               
    - manual                                  # ❌ Do not change
    - automatic                               # ❌ Do not change
  one_cam_at_time: ${collect.one_cam_at_time} # ❌ Do not change
  in_ram: False                               # ❌ Do not change

save:
  raw: false           # ❌ Do not change 
  postprocessed: true  # ❌ Do not change
```

- Manual:
Press space to capture images, and Q to quit.

- Automatic:
The collector starts acquiring images sequentially as soon at least 1 keypoint is visible. Press Q to stop the collection.


### Results

Results are stored in `results/[intrinsics/extrinsics/global]`. Extrinsic and intrinsics data of cameras are stored in the npy files `D,K,euler,position`.

In the `colormaps` folder, there are `.html` files that are plots relative to each camera. They represent, on the location of reprojected points, the distance with respect to the charuco point detected on the image. In this way, you can figure out if the reprojection error is higher on some areas of the image (typically image borders when having high barrel distortion).

The program will also save a Blender scene of the calibrated multi-camera system. As in Blender we can only have pinhole-camera models with no distortion and same focal length on the two axes (perfectly square pixels), the cameras are inserted considering undistorted images with a single focal length value obtained by averaging fx and fy. In the Blender scene, both cameras and charuco boards will be created. Charuco boards, will also have different poses in different time instants, considering different captured images.

