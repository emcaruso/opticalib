# cd to rootpath
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR/.."

# run main
# uv run ./src/main.py --config-path ../configs --config-name cam_calib.yaml python ./src/main.py --config-path ../configs --config-name cam_calib.yaml
python ./src/main.py --config-path ../configs --config-name cam_calib.yaml get_charuco_images=False collect.do=True calibration.do=False
