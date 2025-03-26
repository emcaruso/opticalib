import os, sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from omegaconf import DictConfig, OmegaConf
from objects.charuco import CharucoObject
from utils_ema.image import Image



def main():

    root = Path(os.path.dirname(os.path.realpath(__file__)))
    dir = root / ".."/ "configs" / "objects"
    res_dir = root / ".." / "results" / "charuco_boards"

    os.chdir(str(dir))

    # Create a root window (it will not be shown)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select a file
    yaml_path = Path(filedialog.askopenfilename(title="Select a file", filetypes=[("All files", "*.yaml")]))
    cfg = OmegaConf.load(str(yaml_path))

    #  generate images
    if cfg.type == "charuco":
        charuco = CharucoObject.init_base(cfg)
        images = charuco.generate_charuco_images()
        for i, image in enumerate(images):
            import ipdb; ipdb.set_trace()
            image.save(res_dir / yaml_path.stem / f"charuco_board_{i}.png")


    # show
    Image.show_multiple_images(images)


if __name__ == "__main__":
    main()
