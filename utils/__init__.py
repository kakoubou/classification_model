"""
The `utils` package provides modular tools for data preprocessing, model training, and evaluation.

Modules:
- augment: Defines image augmentation pipelines using imgaug.
- dataset: Custom PyTorch dataset class with integrated imgaug support.
- train_utils: Contains training utility functions, such as fit_one_epoch.
- evaluate: Functions for testing models and visualizing evaluation results.

Example usage:
    from utils import seq, ImgAugmentedDataset, fit_one_epoch, test_model
"""


from .augment import seq
from .dataset import ImgAugmentedDataset
from .train_utils import fit_one_epoch
from .evaluate import test_model


__all__ = [
    "seq",
    "ImgAugmentedDataset",
    "fit_one_epoch",
    "test_model",
]



