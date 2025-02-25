from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import torch
from utils_ema.image import Image
from utils_ema.geometry_pose import Pose


class Features(ABC):
    @property
    @abstractmethod
    def points(self) -> List[torch.Tensor]:
        """numpy array containing a list of 2D point batches, each element for each board."""
        pass

    @property
    @abstractmethod
    def ids(self) -> List[torch.Tensor]:
        """numpy array containing a list of id batches, each element for each board."""
        pass


class ObjectDetector(ABC):

    @abstractmethod
    def detect_features(self, images: List[Image]) -> List[Features]:
        """
        Takes a list of images as input, and returns a list of detected features
        """
        pass

    @abstractmethod
    def draw_features(self, images: List[Image]) -> List[Image]:
        """
        Takes a list of images as input, and returns a list of images
        Each image has the detected features drawn on it
        """
        pass


class Object(ABC):

    @abstractmethod
    def points(self) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def ids(self) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def clone(self, same_pose: bool, same_relative_poses: bool):
        pass

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def put_blender_obj_in_scene(self):
        pass

    #
    # @property
    # @abstractmethod
    # def detector(self) -> ObjectDetector:
    #     pass
