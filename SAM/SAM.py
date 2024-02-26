import torch
from abc import ABC, abstractmethod


class SAM(ABC):
    def __init__(self, name, device=None):
        self.name = name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def set_image(self, image):
        """

        Args:
            image: ndarray, RGB

        Returns: None

        """
        pass

    @abstractmethod
    def box(self, bbox):
        """

        Args:
            bbox: Bbox

        Returns: tuple of
            - mask represented by ndarray of the size of the image with True/False values
            - score

        """
        pass
