# cd to rootpath SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR/.."

# run main
sh ./calibrate_intrinsics.sh
sh ./collect_intrinsics.sh
