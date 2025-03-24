# cd to rootpath SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR/.."

# run main
sh ./scripts/calibrate.sh
sh ./scripts/collect.sh
