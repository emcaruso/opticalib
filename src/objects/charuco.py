import os
import cv2
import numpy as np
import torch
from typing import List, Dict
from omegaconf import DictConfig
from objects.object import Object, Features, ObjectDetector
from utils_ema.charuco import Charuco
from utils_ema.geometry_pose import Pose
from utils_ema.image import Image


def __first_higher(lst, value):
    return next((x for x in lst if x > value), None)


class CharucoObject(Object):

    @classmethod
    def init_with_world_poses(cls, cfg: DictConfig):
        n = cfg.number_board
        return cls(
            cfg,
            pose=Pose(euler=torch.zeros(3, dtype=torch.float32)),
            relative_poses=[
                Pose(euler=torch.zeros(3, dtype=torch.float32)) for _ in range(n)
            ],
        )

    def __init__(self, cfg: DictConfig, pose: Pose, relative_poses: List[Pose]) -> None:
        self.cfg = cfg

        self.length_square_real = self.cfg.boards.square_size
        self.length_marker_real = self.cfg.boards.square_size * (
            self.cfg.boards.length_marker / self.cfg.boards.length_square
        )
        self.n_markers = int(
            self.cfg.boards.number_x_square * self.cfg.boards.number_y_square / 2
        )
        self.n_corners = int(
            (self.cfg.boards.number_x_square - 1)
            * (self.cfg.boards.number_y_square - 1)
        )
        self.aruco_dict_size = __first_higher(
            [50, 100, 250, 1000], self.n_markers * self.cfg.boards.number_board
        )
        self.aruco_dictionary = Charuco.get_aruco_dict(
            n=self.cfg.boards.aruco_n, size=self.aruco_dict_size
        )[0]
        self.length_x = self.cfg.boards.number_x_square * self.length_square_real
        self.length_y = self.cfg.boards.number_y_square * self.length_square_real

        self.__get_boards()
        self.pose = pose
        self.relative_poses = relative_poses
        self.points_list = [self.__get_grid() for _ in range(self.n_boards)]
        self.ids_list = torch.cat(
            [
                torch.arange(i * self.n_corners, (i + 1) * self.n_corners)
                for i in range(self.n_boards)
            ]
        )
        self._detector = CharucoDetector(self)

    def __get_params(self):
        pass

    def clone(self, same_pose: bool = False, same_relative_poses: bool = True):
        if same_pose:
            pose = self.pose
        else:
            pose = self.pose.clone()

        if same_relative_poses:
            relative_poses = self.relative_poses
        else:
            relative_poses = [pose.clone() for pose in self.relative_poses]
        return CharucoObject(cfg, pose, relative_poses)

    def __get_boards(self) -> None:
        self.boards = []
        self.n_boards = len(self.cfg.boards.number)

        for i in range(self.cfg.boards.number_board):
            self.boards.append(
                cv2.aruco.CharucoBoard(
                    (
                        self.cfg.boards.number_x_square,
                        self.cfg.boards.number_y_square,
                    ),
                    self.length_square_real,
                    self.length_marker_real,
                    self.aruco_dictionary,
                    np.arange(i * self.n_markers, (i + 1) * self.n_markers),
                )
            )

    def generate_charuco_images(self) -> List[Image]:
        images = []
        for i in range(len(self.boards)):
            images.append(self.generate_charuco_image(board_id=i))
        return images

    def generate_charuco_image(
        self, board_id: int = 0, marginSize: int = 0, marker_pixs: int = 100
    ) -> Image:
        b = self.boards[board_id]
        img = cv2.aruco.CharucoBoard.generateImage(
            b,
            (
                self.cfg.boards.number_x_square * marker_pixs,
                self.cfg.boards.number_y_square * marker_pixs,
            ),
            marginSize=marginSize,
        )
        image = Image(np.expand_dims(img, -1))
        return image

    def __get_grid(self) -> torch.Tensor:
        grid = torch.tensor(
            [
                [
                    [
                        x * self.length_square_real,
                        y * self.length_square_real,
                        0,
                    ]
                    for x in range(self.cfg.number_x_square - 1)
                    # for x in range(self.number_x_square - 2, -1, -1)
                ]
                for y in range(self.cfg.number_y_square - 1)
            ]
        ).reshape(
            (self.cfg.number_x_square - 1) * (self.cfg.number_y_square - 1),
            3,
        )
        grid[..., 0] -= grid[-1, 0] / 2
        grid[..., 1] -= grid[-1, 1] / 2
        grid[..., 1] *= -1
        grid = grid.type(self.pose.location().dtype)
        return grid.to(self.cfg.device)

    @property
    def points(self) -> torch.Tensor:
        points_list = []
        for board_id, points in enumerate(self.points_list):
            # self.pose = w_T_obj
            # self.relative_poses = obj_T_board
            p = (self.pose.rotation() @ points.T).T + self.pose.location()
            points_list.append(p)
        return torch.cat(points_list, dim=0)

    @property
    def ids(self) -> torch.Tensor:
        return self.ids_list

    @property
    def detector(self) -> CharucoDetector:
        return self._detector


class CharucoDetector(ObjectDetector):

    def __init__(self, charuco_obj: CharucoObject) -> None:
        self.obj = charuco_obj
        self.__get_detector()
        # Image.show_multiple_images(self.generate_charuco_images())
        # im = Image.from_path("/home/emcarus/Desktop/git_repos/refactored_project/opticalib/data/tt.png")
        # self.draw_charuco(im).show()

    def __get_detector(self) -> None:
        detector_params = cv2.aruco.DetectorParameters()
        for k, v in self.obj.cfg.detector.items():
            detector_params.__setattr__(k, v)
        self.detector = cv2.aruco.ArucoDetector(
            self.obj.aruco_dictionary, detector_params
        )

    def detect_charuco_corners(self, image: Image) -> Dict[str, List[torch.Tensor]]:

        img = image.uint8().numpy()

        charuco_corners_all = []
        charuco_corners_ids = []
        charuco_markers_all = []
        charuco_markers_ids = []

        marker_corners, marker_ids, _ = self.detector.detectMarkers(img)

        if marker_corners:

            for i, b in enumerate(self.obj.boards):

                valid_corners = []
                valid_ids = []
                valid_range = np.arange(
                    i * self.obj.n_markers, (i + 1) * self.obj.n_markers
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
                        torch.from_numpy(charuco_ids + i * self.obj.n_corners)
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
                    img, charuco_corners[i], charuco_ids[i]
                )
            if markers:
                img = cv2.aruco.drawDetectedMarkers(
                    img, marker_corners[i], marker_ids[i], borderColor
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
