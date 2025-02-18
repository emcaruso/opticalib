from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import torch
from utils_ema.image import Image
from utils_ema.pose import Pose


class Features(ABC):
    @property
    @abstractmethod
    def points(self) -> List[torch.Tensor]:
        """numpy array containing a batch of 2D points."""
        pass

    @property
    @abstractmethod
    def ids(self) -> List[torch.Tensor]:
        """numpy array containing the ids of the 2D points."""
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
    @property
    @abstractmethod
    def parameters(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def pose_list(self) -> List[Pose]:
        pass

    @property
    @abstractmethod
    def detector(self) -> ObjectDetector:
        pass
