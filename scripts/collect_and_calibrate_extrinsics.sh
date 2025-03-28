# cd to rootpath SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR/.."

# run main
sh ./calibrate_extrinsics.sh
sh ./collect_extrinsics.sh
