from abc import ABC, abstractmethod

class Object(ABC):

    @abstractmethod
    def detect_features(self):
        pass

    @abstractmethod
    def draw_features(self, image):
        pass
