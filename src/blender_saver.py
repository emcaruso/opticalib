import bpy
import os
import random
from utils_ema.camera_cv import Camera_cv
from pathlib import Path
from typing import List
from objects.object import Object
from logging import Logger
from utils_ema.blender_utils import set_object_pose, put_cam_in_scene 
from scene import Scene

def blender_save(save_dir: str, scene_calib: Scene, logger: Logger):

    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Link object to the scene
    scene_blender = bpy.context.scene

    o = scene_calib.objects[0].put_blender_obj_in_scene(scene_blender, save_dir)
    for i, obj in enumerate(scene_calib.objects):
        set_object_pose(o, obj.pose)
        o.keyframe_insert(data_path="location", frame=i)
        o.keyframe_insert(data_path="rotation_euler", frame=i)

    for i, cam in enumerate(scene_calib.cameras):
        cam_obj = put_cam_in_scene(scene_blender, name=f"cam_{i:03d}")
        set_object_pose(cam_obj, cam.pose)

    os.makedirs(save_dir, exist_ok=True)
    filepath = Path(save_dir) / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(filepath))


