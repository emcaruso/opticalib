import os
import cv2
from utils_ema.charuco import Charuco

class CharucoObj():

    def __init__(self, path):
        self.path = path


    def get_charuco_board_parameters(self):

        assert os.path.isfile(self.path)

        fs = cv2.FileStorage(self.path, cv2.FILE_STORAGE_READ)
        board_data = {
            "square_size": fs.getNode("square_size").real(),
            "resolution_x": int(fs.getNode("resolution_x").real()),
            "resolution_y": int(fs.getNode("resolution_y").real()),
            "number_x_square": int(fs.getNode("number_x_square").real()),
            "number_y_square": int(fs.getNode("number_y_square").real()),
            "length_square": fs.getNode("length_square").real(),
            "length_marker": fs.getNode("length_marker").real(),
            "length_square_real": fs.getNode("square_size").real(),
            "length_marker_real": fs.getNode("square_size").real()
            * (fs.getNode("length_marker").real() / fs.getNode("length_square").real()),
            "number_board": int(fs.getNode("number_board").real()),
            "aruco_dictionary": Charuco.get_aruco_dict(n=6, size=100)[0],
            "thickness": fs.getNode("thickness").real(),
        }
        fs.release()
        return board_data
