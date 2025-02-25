import os
import bpy
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from typing import List, Dict
from omegaconf import DictConfig
from objects.object import Object, Features, ObjectDetector
from utils_ema.charuco import Charuco
from utils_ema.geometry_pose import Pose
from utils_ema.geometry_euler import eul
from utils_ema.image import Image


def first_higher(lst, value):
    return next((x for x in lst if x > value), None)


class CharucoObject(Object):

    @classmethod
    def init_base(cls, cfg: DictConfig):
        n = cfg.boards.n_boards
        p = CharucoObject.__get_board_params(cfg)
        return cls(
            cfg=cfg,
            params=p,
            pose=Pose(euler=eul(torch.zeros(3, dtype=torch.float32))),
            relative_poses=[
                Pose(euler=eul(torch.zeros(3, dtype=torch.float32))) for _ in range(n)
            ],
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
        self.params.ids_list = self.params.ids_list.to(device)
        return self

    @staticmethod
    def __get_board_params(cfg):
        params = SimpleNamespace(**cfg.boards)
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
            CharucoObject.__get_grid(params) for _ in range(params.n_boards)
        ]
        params.ids_list = torch.cat(
            [
                torch.arange(i * params.n_corners, (i + 1) * params.n_corners)
                for i in range(params.n_boards)
            ]
        )
        params.detector_params = cfg.detector
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
        return CharucoObject(self.cfg, self.params, pose, relative_poses)

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
    def __get_grid(params) -> torch.Tensor:
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
        return grid

    def points(self) -> List[torch.Tensor]:
        points_list = []
        for board_id, points in enumerate(self.params.points_list):
            R1 = self.pose.rotation()
            R2 = self.relative_poses[board_id].rotation()
            t1 = self.pose.location()
            t2 = self.relative_poses[board_id].location()
            R = R2 @ R1  # Combine rotations
            t = R2 @ t1 + t2  # Combine translations
            p = points @ R.T + t
            points_list.append(p)
        return points_list

    def ids(self) -> List[torch.Tensor]:
        return self.params.ids_list

    def put_blender_obj_in_scene(self, scene):

        # --- Step 2: Create a new plane mesh object ---
        mesh = bpy.data.meshes.new(name="PlaneMesh")  # Create an empty mesh
        obj = bpy.data.objects.new(
            name="Plane", object_data=mesh
        )  # Create an object using the mesh

        # Create a plane geometry and assign it to the mesh
        vertices = [  # Define the 4 vertices of the plane
            (-1, -1, 0),
            (1, -1, 0),
            (1, 1, 0),
            (-1, 1, 0),
        ]
        faces = [(0, 1, 2, 3)]  # Define the 4 faces of the plane
        mesh.from_pydata(vertices, [], faces)

        scene.collection.objects.link(obj)

    # # Create a new mesh and object
    # mesh = bpy.data.meshes.new(name="PlaneMesh")
    # obj = bpy.data.objects.new(name="PlaneObject", object_data=mesh)
    # scene.collection.objects.link(obj)
    #
    # # Create boards
    # boards = []
    # for board_id, p in enumerate(self.relative_poses):
    #     bpy.ops.mesh.primitive_plane_add(size=1)
    #     board = bpy.context.active_object
    #     board.name = f"board_{board_id:03d}"
    #     return board
    #     boards.append(board)
    #
    # # make boards children of an empty object
    # bpy.ops.object.empty_add(type="PLAIN_AXES")
    # empty = bpy.context.active_object
    # empty.name = "calib_obj"
    # for board in boards:
    #     board.parent = empty
    #
    # return empty


class CharucoDetector(ObjectDetector):

    def __init__(self, params: SimpleNamespace) -> None:
        self.params = params
        self.__get_detector()

    def __get_detector(self) -> None:
        detector_params = cv2.aruco.DetectorParameters()
        for k, v in self.params.detector_params.items():
            detector_params.__setattr__(k, v)
        self.detector = cv2.aruco.ArucoDetector(
            self.params.aruco_dictionary, detector_params
        )

    def detect_charuco_corners(self, image: Image) -> Dict[str, List[torch.Tensor]]:

        img = image.uint8().numpy()

        charuco_corners_all = []
        charuco_corners_ids = []
        charuco_markers_all = []
        charuco_markers_ids = []

        marker_corners, marker_ids, _ = self.detector.detectMarkers(img)

        if marker_corners:

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

                retval, charuco_corners, charuco_ids = (
                    cv2.aruco.interpolateCornersCharuco(
                        valid_corners, valid_ids, img, b
                    )
                )
                if retval:
                    charuco_corners_all.append(torch.from_numpy(charuco_corners))
                    charuco_corners_ids.append(
                        torch.from_numpy(charuco_ids + i * self.params.n_corners)
                    )
                    charuco_markers_all.append(torch.from_numpy(valid_corners))
                    charuco_markers_ids.append(torch.from_numpy(valid_ids))

            # if len(charuco_corners_all) == 0:torch.from_numpy()
            #     return {
            #         "charuco_corners": np.array([]),
            #         "charuco_ids": np.array([]),
            #         "marker_corners": np.array([]),
            #         "marker_ids": np.array([]),
            #     }
            #     return np.array([]), np.array([]), np.array([]), np.array([])

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

    def draw_features(self, images) -> List[Image]:
        return [self.draw_charuco(i, corners=True, markers=False) for i in images]

    def detect_features(self, images) -> List[Features]:
        features = [self.detect_charuco_corners(i) for i in images]
        return [
            CharucoFeatures(corners=f["charuco_corners"], ids=f["charuco_ids"])
            for f in features
        ]


class CharucoFeatures(Features):

    def __init__(self, corners: List[torch.Tensor], ids: List[torch.Tensor]):
        self.charuco_corners = corners
        self.charuco_ids = ids

    @property
    def points(self) -> List[torch.Tensor]:
        return self.charuco_corners

    @property
    def ids(self) -> List[torch.Tensor]:
        return self.charuco_ids
