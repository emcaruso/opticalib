# cd to rootpath SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR/.."

# run main
sh ./scripts/calibrate_global.sh
sh ./scripts/collect_global.sh
