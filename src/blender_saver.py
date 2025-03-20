import bpy
import os
import random
from utils_ema.camera_cv import Camera_cv
from pathlib import Path
from typing import List
from objects.object import Object
from logging import Logger
from utils_ema.blender_utils import set_object_pose, put_cam_in_scene , set_background_images
from utils_ema.geometry_pose import Pose
from utils_ema.geometry_euler import eul
from utils_ema.camera_cv import Camera_cv
from scene import Scene, Intrinsics


def blender_save(save_dir: str, scene_calib: Scene, logger: Logger):

    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Link object to the scene
    scene_blender = bpy.context.scene

    # objects
    o = scene_calib.objects.put_blender_obj_in_scene(scene_blender, save_dir)
    for time_id in range(scene_calib.time_instants):
        position = scene_calib.objects.pose.position.reshape(-1,3)[time_id]
        euler = scene_calib.objects.pose.euler.e.reshape(-1,3)[time_id]
        pose = Pose(position=position, euler=eul(euler))
        set_object_pose(o, pose)
        o.keyframe_insert(data_path="location", frame=time_id)
        o.keyframe_insert(data_path="rotation_euler", frame=time_id)

    # cameras
    for cam_id, cam in enumerate(scene_calib.cameras):
        cam_obj = put_cam_in_scene(scene_blender, cam)
        set_background_images(cam_obj, Path(save_dir) / "undist_images" / f"cam_{cam_id:03d}")
        # set_object_pose(cam_obj, cam.pose)
        # blender_camera_transform(cam_obj)
        # cam_obj.keyframe_insert(data_path="location", frame=0)
        # cam_obj.keyframe_insert(data_path="rotation_euler", frame=0)
        #
    os.makedirs(save_dir, exist_ok=True)
    filepath = Path(save_dir) / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(filepath))


