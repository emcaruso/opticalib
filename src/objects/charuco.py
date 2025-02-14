import os
import cv2
import numpy as np
from typing import List, Tuple
from omegaconf import DictConfig
from objects.object import Object
from utils_ema.charuco import Charuco
from utils_ema.image import Image


class CharucoBoard():
    pass

def first_higher(lst, value):
    return next((x for x in lst if x > value), None)

class CharucoObjects(Object):
    

    def __init__(self, cfg : DictConfig):
        self.cfg = cfg

        self.length_square_real = self.cfg.boards.square_size
        self.length_marker_real = self.cfg.boards.square_size * (self.cfg.boards.length_marker / self.cfg.boards.length_square)
        self.n_markers = int(self.cfg.boards.number_x_square * self.cfg.boards.number_y_square / 2)
        self.n_corners = int((self.cfg.boards.number_x_square - 1) * (self.cfg.boards.number_y_square - 1))
        self.aruco_dict_size = first_higher([50, 100, 250, 1000], self.n_markers * self.cfg.boards.number_board)
        self.aruco_dictionary = Charuco.get_aruco_dict(n=self.cfg.boards.aruco_n, size=self.aruco_dict_size)[0]
        self.length_x = self.cfg.boards.number_x_square * self.length_square_real
        self.length_y = self.cfg.boards.number_y_square * self.length_square_real
        self.__get_boards()
        self.__get_detector()
        # Image.show_multiple_images(self.generate_charuco_images())
        # im = Image.from_path("/home/emcarus/Desktop/git_repos/refactored_project/opticalib/data/tt.png")
        # self.draw_charuco(im).show()

    def __get_boards(self) -> None:
        self.boards = []

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
                np.arange( i * self.n_markers, (i+1) * self.n_markers)
                )
            )

    def __get_detector(self) -> None:
        detector_params = cv2.aruco.DetectorParameters()
        for k,v in self.cfg.detector.items():
            detector_params.__setattr__(k, v)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, detector_params)

    def generate_charuco_images(self) -> List[Image]:
        images = []
        for i in range(len(self.boards)):
            images.append(self.generate_charuco_image(board_id=i))
        return images

    def generate_charuco_image(self, board_id : int = 0, marginSize : int = 0, marker_pixs : int = 100):
        b = self.boards[board_id]
        img = cv2.aruco.CharucoBoard.generateImage(
            b,
            (self.cfg.boards.number_x_square * marker_pixs, self.cfg.boards.number_y_square * marker_pixs),
            marginSize=marginSize,
        )
        image = Image(np.expand_dims(img, -1))
        return image


    def detect_charuco_corners(self, image : Image ) -> Tuple[np.ndarray]:

        img = image.uint8().numpy()

        charuco_corners_all = []
        charuco_corners_ids = []
        charuco_markers_all = []
        charuco_markers_ids = []

        marker_corners, marker_ids, _ = self.detector.detectMarkers(img)

        if marker_corners:

            for i, b in enumerate(self.boards):

                valid_corners = []
                valid_ids = []
                valid_range = np.arange( i * self.n_markers, (i+1) * self.n_markers)

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
                    charuco_corners_all.append(charuco_corners)
                    charuco_corners_ids.append(charuco_ids+ i * self.n_corners)
                    charuco_markers_all.append(valid_corners)
                    charuco_markers_ids.append(valid_ids)

            if len(charuco_corners_all) == 0:
                return np.array([]), np.array([]), np.array([]), np.array([])


        return (
            charuco_corners_all,
            charuco_corners_ids,
            charuco_markers_all,
            charuco_markers_ids,
        )

    def detect_charuco_corners_multi(self, images):
        R = [[], [], [], []]

        for image in images:
            res = self.detect_charuco_corners(image)
            for i, r in enumerate(res):
                if len(r) != 0:
                    R[i].append(r)

        for i, r in enumerate(R):
            if len(R[i]) != 0:
                R[i] = np.vstack(R[i])

        return tuple(R)

    def draw_charuco(self, image, corners=True, markers=True, borderColor=(255, 0, 0)):
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            self.detect_charuco_corners(image)
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

    def draw_features(self, images):
        return [ self.draw_charuco(i, corners=True, markers=False) for i in images ]

    def detect_features(self, images):
        return self.detect_charuco_corners()


