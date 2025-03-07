import os
import bpy
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from typing import List, Dict
from omegaconf import DictConfig
from utils_ema.charuco import Charuco
from utils_ema.geometry_pose import Pose
from utils_ema.geometry_euler import eul
from utils_ema.image import Image
from utils_ema.blender_utils import put_plane_in_scene, set_object_pose, set_object_texture


def first_higher(lst, value):
    return next((x for x in lst if x > value), None)

class CharucoFeatures():

    def __init__(self, corners: List[torch.Tensor], ids: List[torch.Tensor], device: str) -> None:
        self.charuco_corners = corners
        self.charuco_ids = ids
        self.device = device

    @property
    def points(self) -> List[torch.Tensor]:
        return self.charuco_corners

    @property
    def ids(self) -> List[torch.Tensor]:
        return self.charuco_ids

    def to(self, device):
        self.dedvice = device
        self.charuco_corners = [c.to(device) for c in self.charuco_corners]
        self.charuco_ids = [i.to(device) for i in self.charuco_ids]

class CharucoObject():

    @classmethod
    def init_base(cls, cfg: DictConfig, device="cpu"):
        n = cfg.n_boards
        p = CharucoObject.__get_board_params(cfg, device=device)
        return cls(
            cfg=cfg,
            params=p,
            pose=Pose(position=torch.zeros(3, dtype=torch.float32), euler=eul(torch.zeros(3, dtype=torch.float32))),
            relative_poses=[
                Pose(position=torch.zeros(3, dtype=torch.float32), euler=eul(torch.zeros(3, dtype=torch.float32))) for _ in range(n)
            ],
            device = device
        )

    def __init__(
        self,
        cfg: DictConfig,
        params: SimpleNamespace,
        pose: Pose,
        relative_poses: List[Pose],
        device="cpu",
    ) -> None:
        self.cfg = cfg
        self.params = params
        self.pose = pose
        self.relative_poses = relative_poses
        self.device = device
        self.to(device)

    def to(self, device):
        self.pose.to(device)
        for p in self.relative_poses:
            p.to(device)
        self.params.points_list = [p.to(device) for p in self.params.points_list]
        self.params.ids_list = [i.to(device) for i in self.params.ids_list]
        return self

    @staticmethod
    def __get_board_params(cfg, device="cpu"):
        params = SimpleNamespace(**cfg)
        params.aspect_ratio = params.number_x_square / params.number_y_square
        params.length_square_real = params.square_size
        params.length_marker_real = params.square_size * (
            params.length_marker / params.length_square
        )
        params.n_markers = int(params.number_x_square * params.number_y_square / 2)
        params.n_corners = int(
            (params.number_x_square - 1) * (params.number_y_square - 1)
        )
        params.aruco_dict_size = first_higher(
            [50, 100, 250, 1000], params.n_markers * params.n_boards
        )
        params.aruco_dictionary = Charuco.get_aruco_dict(
            n=params.aruco_n, size=params.aruco_dict_size
        )[0]
        params.length_x = params.number_x_square * params.length_square_real
        params.length_y = params.number_y_square * params.length_square_real
        CharucoObject.__put_board_cv2(params)
        params.points_list = [
            CharucoObject.__get_grid(params, device) for _ in range(params.n_boards)
        ]
        params.ids_list = [
                torch.arange(i * params.n_corners, (i + 1) * params.n_corners, device=device)
                for i in range(params.n_boards)
            ]
        
        return params

    @staticmethod
    def __put_board_cv2(params) -> None:
        params.boards_cv2 = []

        for i in range(params.n_boards):
            params.boards_cv2.append(
                cv2.aruco.CharucoBoard(
                    (
                        params.number_x_square,
                        params.number_y_square,
                    ),
                    params.length_square_real,
                    params.length_marker_real,
                    params.aruco_dictionary,
                    np.arange(i * params.n_markers, (i + 1) * params.n_markers),
                )
            )

    def clone(self, same_pose: bool, same_relative_poses: bool):
        if same_pose:
            pose = self.pose
        else:
            pose = self.pose.clone()

        if same_relative_poses:
            relative_poses = self.relative_poses
        else:
            relative_poses = [pose.clone() for pose in self.relative_poses]
        return CharucoObject(self.cfg, self.params, pose, relative_poses, device=self.device)

    def generate_charuco_images(self) -> List[Image]:
        images = []
        for i in range(len(self.params.boards_cv2)):
            images.append(self.generate_charuco_image(board_id=i))
        return images

    def generate_charuco_image(
        self, board_id: int = 0, marginSize: int = 0, marker_pixs: int = 100
    ) -> Image:
        b = self.params.boards_cv2[board_id]
        img = cv2.aruco.CharucoBoard.generateImage(
            b,
            (
                self.params.number_x_square * marker_pixs,
                self.params.number_y_square * marker_pixs,
            ),
            marginSize=marginSize,
        )
        image = Image(np.expand_dims(img, -1))
        return image

    @staticmethod
    def __get_grid(params, device = "cpu") -> torch.Tensor:
        grid = torch.tensor(
            [
                [
                    [
                        x * params.length_square_real,
                        y * params.length_square_real,
                        0,
                    ]
                    for x in range(params.number_x_square - 1)
                    # for x in range(self.number_x_square - 2, -1, -1)
                ]
                for y in range(params.number_y_square - 1)
            ]
        ).reshape(
            (params.number_x_square - 1) * (params.number_y_square - 1),
            3,
        )
        grid[..., 0] -= grid[-1, 0] / 2
        grid[..., 1] -= grid[-1, 1] / 2
        grid[..., 1] *= -1
        return grid.to(device)

    def points(self) -> List[torch.Tensor]:
        points_list = []
        for board_id, points in enumerate(self.params.points_list):
            R1 = self.pose.rotation()
            t1 = self.pose.location()
            R2 = self.relative_poses[board_id].rotation()
            t2 = self.relative_poses[board_id].location()
            R = R2 @ R1  # Combine rotations
            t = R2 @ t1 + t2  # Combine translations
            p = points @ R.T + t
            points_list.append(p)
        return points_list

    def ids(self) -> List[torch.Tensor]:
        return self.params.ids_list

    def put_blender_obj_in_scene(self, scene, scene_dir):

        images = self.generate_charuco_images()
        img_paths = []
        for i, img in enumerate(images):
            img_paths.append( str(Path(scene_dir) / "charuco_images" / f"board_{i:03d}.png") )
            img.save(img_paths[-1])

        # # Create boards
        boards = []
        for board_id, p in enumerate(self.relative_poses):
            name = f"board_{board_id:03d}"
            board = put_plane_in_scene(scene, name, self.params.length_x, self.params.length_y)
            set_object_pose(board, p)
            set_object_texture(name=name, obj=board, image_path=img_paths[board_id])
            boards.append(board)

        # make boards children of an empty object
        bpy.ops.object.empty_add(type="PLAIN_AXES")
        empty = bpy.context.active_object
        empty.name = "calib_obj"
        for board in boards:
            board.parent = empty

        return empty


class CharucoDetector():

    def __init__(self, params: SimpleNamespace) -> None:
        self.params = params
        self.__get_detector()

    def __get_detector(self) -> None:
        detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.params.aruco_dictionary, detector_params
        )

    def images_has_at_least_one_feature(self, images : List[Image]) -> bool:
        features = self.detect_features(images)
        for fts in features:
            for c in fts.charuco_corners:
                if len(c) > 0:
                    return True
        return False

    def detect_charuco_corners(self, image: Image, device:str = "cpu") -> Dict[str, List[torch.Tensor]]:

        img = image.uint8().numpy()

        charuco_corners_all = []
        charuco_corners_ids = []
        charuco_markers_all = []
        charuco_markers_ids = []

        marker_corners, marker_ids, _ = self.detector.detectMarkers(img)
        if marker_ids is None:
            marker_ids = []
            marker_corners = []

        for i, b in enumerate(self.params.boards_cv2):

            valid_corners = []
            valid_ids = []
            valid_range = np.arange(
                i * self.params.n_markers, (i + 1) * self.params.n_markers
            )

            for j, id in enumerate(marker_ids):
                if id in valid_range:
                    valid_corners.append(marker_corners[j])
                    valid_ids.append(id)
            valid_corners = np.array(valid_corners)
            valid_ids = np.array(valid_ids)


            if len(valid_corners) == 0:
                valid_corners = np.empty([0,1,4,2])
                valid_ids = np.empty([0,1])
                retval = False
            else:
                retval, charuco_corners, charuco_ids = (
                    cv2.aruco.interpolateCornersCharuco(
                        valid_corners, valid_ids, img, b
                    )
                )
            if not retval:
                charuco_corners = np.empty([0,1,2])
                charuco_ids = np.empty([0,1])


            charuco_corners_all.append(torch.tensor(charuco_corners, device=device))
            charuco_corners_ids.append(
                torch.tensor(charuco_ids + i * self.params.n_corners, device=device)
            )
            charuco_markers_all.append(torch.tensor(valid_corners, device=device ))
            charuco_markers_ids.append(torch.tensor(valid_ids, device=device))


        return {
            "charuco_corners": charuco_corners_all,
            "charuco_ids": charuco_corners_ids,
            "marker_corners": charuco_markers_all,
            "marker_ids": charuco_markers_ids,
        }

    def draw_charuco(
        self, image, corners=True, markers=True, borderColor=(255, 0, 0)
    ) -> Image:


        charuco_corners, charuco_ids, marker_corners, marker_ids = list(
            self.detect_charuco_corners(image).values()
        )

        if len(charuco_corners) == 0:
            return image.clone()

        img = image.uint8().numpy().copy()
        for i in range(len(charuco_ids)):
            if corners:
                img = cv2.aruco.drawDetectedCornersCharuco(
                    img, charuco_corners[i].numpy(), charuco_ids[i].numpy()
                )
            if markers:
                img = cv2.aruco.drawDetectedMarkers(
                    img, marker_corners[i].numpy(), marker_ids[i].numpy(), borderColor
                )
        return Image(img)

    def draw_features(self, images: List[Image]) -> List[Image]:
        return [self.draw_charuco(i, corners=True, markers=False) for i in images]

    def detect_features(self, images: List[Image], device = "cpu") -> List[CharucoFeatures]:
        features = [self.detect_charuco_corners(i, device=device) for i in images]
        return [
            CharucoFeatures(corners=f["charuco_corners"], ids=f["charuco_ids"])
            for f in features
        ]


