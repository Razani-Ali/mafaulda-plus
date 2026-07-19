from typing import Callable, Tuple, Optional, Any, Optional, List, Union, Dict
from .transform import FewShotSampler, VirtualFeatureWindow
from .dataset_frameworks import PyTorchMafauldaDataset, TFMafauldaGenerator
import numpy as np


def sample_few_shot_features(
    X_base: np.ndarray,
    Y_base: np.ndarray,
    numeric_to_string: Dict[int, str],
    valid_folds: Optional[List[int]] = None,
    valid_files: Optional[List[int]] = None,
    meta_base: Optional[Tuple[np.ndarray, np.ndarray]] = (None, None),
    seed: Optional[int] = None
) -> FewShotSampler:
    """
    Samples isolated task frames tailored for Few-Shot / Meta-Learning configurations.
    Enforces strict structural isolation and maps indices lazily over disk grids.

    Args:
        X_base (np.ndarray): Extracted features 4D+ tensor [Folds, Files, Windows, *Dims].
        Y_base (np.ndarray): Broadcasted string labels array [Folds, Files, Windows].
        numeric_to_string (Dict[int, str]): Map dictionary translating integers to strings.
        valid_folds (List[int], optional): Specific folds allocated for task extraction.
        valid_files (List[int], optional): Subset array of allowed structural file indices.
        meta_base (Tuple[np.ndarray, np.ndarray]): Broadcasted metadata (Severity, RPM).
        seed (int, optional): Fixed user-defined random seed ensuring reproducibility.

    Returns:
        FewShotSampler: An initialized sampler object ready to generate tasks.
    """
    sampler = FewShotSampler(
        X_base=X_base, Y_base=Y_base, numeric_to_string=numeric_to_string,
        valid_folds=valid_folds, valid_files=valid_files, meta_base=meta_base,
        seed=seed
    )
    return sampler


def get_pytorch_feature_loader(
    X_base: np.ndarray,
    Y_base: np.ndarray,
    meta_base: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    class_to_idx: Dict[str, int],
    batch_size: int = 32,
    shuffle: bool = True,
    valid_folds: Optional[List[int]] = None,
    valid_files: Optional[List[int]] = None,
    **dataloader_kwargs
):
    """
    Assembles a high-performance, native PyTorch DataLoader wrapping the pre-windowed tensor.
    Imports PyTorch lazily to avoid heavy framework initialization overheads.

    Args:
        X_base (np.ndarray): Extracted features 4D+ tensor.
        Y_base (np.ndarray): Broadcasted string labels array.
        meta_base (Tuple): Parallel broadcasted metadata arrays (Severity, RPM).
        class_to_idx (Dict[str, int]): Dictionary translating categorical names into integers.
        batch_size (int): Minibatch density streamed on each iterative step.
        shuffle (bool): Randomizes batch sequence orderings if True.
        valid_folds (List[int], optional): Targeted cross-validation subset regions.
        valid_files (List[int], optional): Subset array of allowed structural file indices.
        **dataloader_kwargs: Arbitrary keyword options fed directly to PyTorch.

    Returns:
        torch.utils.data.DataLoader: A native, zero-copy loader yielding batches of features and labels.
    """
    from torch.utils.data import DataLoader
    
    vw = VirtualFeatureWindow(
        X=X_base, Y=Y_base, meta_base=meta_base,
        valid_folds=valid_folds, valid_files=valid_files
    )
    wrapper = PyTorchMafauldaDataset(virtual_window=vw, class_to_idx=class_to_idx)
    return DataLoader(wrapper.get_dataset(), batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)


def get_tensorflow_featureset(
    X_base: np.ndarray,
    Y_base: np.ndarray,
    meta_base: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    class_to_idx: Dict[str, int],
    batch_size: int = 32,
    valid_folds: Optional[List[int]] = None,
    valid_files: Optional[List[int]] = None,
):
    """
    Builds an optimized tf.data.Dataset pipeline connected directly to the feature matrix.
    Uses standard output signatures to guarantee safe execution across TF environments.

    Args:
        X_base (np.ndarray): Extracted features 4D+ tensor.
        Y_base (np.ndarray): Broadcasted string labels array.
        meta_base (Tuple): Parallel broadcasted metadata arrays (Severity, RPM).
        class_to_idx (Dict[str, int]): Code map dictionary pointing category strings to integer ids.
        batch_size (int): Number of arrays packaged inside each parallel streaming batch.
        valid_folds (List[int], optional): Authorized cross-validation temporal folds.
        valid_files (List[int], optional): Subset array of allowed structural file indices.

    Returns:
        tf.data.Dataset: A high-throughput pre-batched and pre-fetched tensorflow stream instance.
    """
    vw = VirtualFeatureWindow(
        X=X_base, Y=Y_base, meta_base=meta_base,
        valid_folds=valid_folds, valid_files=valid_files
    )
    wrapper = TFMafauldaGenerator(virtual_window=vw, class_to_idx=class_to_idx)
    return wrapper.get_dataset(batch_size=batch_size)