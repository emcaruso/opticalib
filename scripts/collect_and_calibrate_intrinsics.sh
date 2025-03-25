# cd to rootpath SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR/.."

# run main
sh ./scripts/calibrate_intrinsics.sh
sh ./scripts/collect_intrinsics.sh
