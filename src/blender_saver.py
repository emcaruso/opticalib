import bpy
import os
import random
from utils_ema.camera_cv import Camera_cv
from pathlib import Path
from typing import List
from objects.object import Object
from logging import Logger
from utils_ema.blender_utils import set_object_pose

def blender_save(
    save_dir: str, objects: List[Object], cameras: List[Camera_cv], logger: Logger
):

    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Link object to the scene
    scene = bpy.context.scene

    for obj in objects:
        o = obj.put_blender_obj_in_scene(scene, save_dir)
        # o.name =
        break

    os.makedirs(save_dir, exist_ok=True)
    filepath = Path(save_dir) / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(filepath))

    #
    # for cam in cameras:
    #     print(cam)
    # pass


