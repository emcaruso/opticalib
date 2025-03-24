# cd to rootpath
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR/.."

# run main
# python ./src/main.py --config-path ../configs --config-name cam_calib.yaml get_charuco_images=True collect.do=False calibration.do=False
uv run ./src/main.py --config-path ../configs --config-name cam_calib.yaml get_charuco_images=True collect.do=False calibration.do=False
