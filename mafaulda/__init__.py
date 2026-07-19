import numpy as np
from typing import Dict, Callable, List, Tuple, Union, Optional
import os
from .downloader import SecureResumableDownloader
from .ingestion import MAFAULDAIngestor
from .filtering import ParallelZarrFilterEngine
from .dataset import MAFAULDARawLoader, PhysicalSlidingWindow, VirtualSlidingWindow, generate_stratified_file_split
from .dataset_frameworks import PyTorchMafauldaDataset, TFMafauldaGenerator
from .fewshot_sampler import FewShotSampler
import tempfile
from .transform import ZeroRAMFeatureWorkspace, load_zarr_to_tensor


def download(
    url: str = "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/full.zip",
    target_path: str = "data/MAFAULDA.zip",
    min_size_bytes: int = 12 * 1024 * 1024 * 1024,
    max_retries: int = 10,
    retry_delay: float = 3.0,
    chunk_size: int = 1024 * 1024,
    timeout: int = 30,
    replace: bool = False,
    num_connections: int = 8
) -> bool:
    """
    Downloads the MAFAULDA dataset archive with multi-threaded chunking and automatic resume.

    Args:
        url (str): The direct URL of the MAFAULDA database zip file.
        target_path (str): The local file path where the downloaded ZIP file will be saved.
        min_size_bytes (int): Minimum acceptable file size acting as a corruption gate.
        max_retries (int): Maximum network retry loops before failure.
        retry_delay (float): Time gap (in seconds) to wait before re-opening a dropped socket.
        chunk_size (int): Stream buffer size in bytes for low-level I/O disk writes.
        timeout (int): Network connection and read timeout limit in seconds.
        replace (bool): If True, overrides any existing file at target_path.
        num_connections (int): Number of parallel connection threads for Range Requests.

    Returns:
        bool: True if the download was fully successful and validated, False otherwise.
    """
    downloader = SecureResumableDownloader(
        url=url, target_path=target_path, min_size_bytes=min_size_bytes,
        max_retries=max_retries, retry_delay=retry_delay, chunk_size=chunk_size,
        timeout=timeout, replace=replace, num_connections=num_connections
    )
    return downloader.download()


def ingest(
    raw_data_dir: str,
    zarr_store_path: str,
    max_workers: int = 8,
    dtype: np.dtype = np.float16
) -> None:
    """
    Parses hierarchical raw MAFAULDA CSV files, extracts structured metadata, 
    and compresses the signals into a single, high-performance Zarr database.

    Args:
        raw_data_dir (str): The directory path containing the unzipped nested folder architecture.
        zarr_store_path (str): The target directory path where the binary Zarr store will be built.
        max_workers (int): The number of parallel CPU worker processes allocated.
        dtype (numpy.dtype): The numerical precision used to convert and cast the raw text data.
    """
    ingestor = MAFAULDAIngestor(
        raw_data_dir=raw_data_dir, zarr_store_path=zarr_store_path,
        max_workers=max_workers, dtype=dtype
    )
    ingestor.ingest_all()


def filter(
    src_store_path: str,
    dst_store_path: str,
    filter_map: Dict[str, Callable],
    downsample_factor: int = 1,
    max_workers: int = 8
) -> None:
    """
    Concurrently processes an existing MAFAULDA Zarr database by executing user-injected 
    filters channel-by-channel and optionally downsampling the signal lengths.
    """
    engine = ParallelZarrFilterEngine(
        src_store_path=src_store_path,
        dst_store_path=dst_store_path,
        filter_map=filter_map,
        downsample_factor=downsample_factor,
        max_workers=max_workers
    )
    engine.run()


def load(
    zarr_path: str,
    folds: int = 4,
    labeling_strategy: str = 'only types',
    group_misalignment: bool = False,
    use_memmap: bool = True,
    memmap_path: Optional[str] = None,
    selected_sensors: List[str] = 
                    ['UH Axial Acc', 'UH Radial Acc', 'UH Tangential Acc',
                     'OH Axial Acc', 'OH Radial Acc', 'OH Tangential Acc'],
    target_classes: Union[str, List[str]] = 'all',
    target_severities: Union[str, List[str]] = 'all',
    rpm_range: Union[str, Tuple[float, float]] = 'all',
    max_workers: int = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts, filters, and parallel-populates an optimized 4D tensor structured for ML.
    The data layout is pre-arranged as C-contiguous arrays mapping [Folds, Files, Channels, Chunk_Length].

    Args:
        zarr_path (str): Path to the compiled source Zarr directory store.
        folds (int): Cross-validation stratification folds slicing the contiguous timeline. Defaults to 1.
        labeling_strategy (str): 'only types' or 'types & severity' configurations.
        group_misalignment (bool): If True, maps structural directional flaws under a generic category.
        use_memmap (bool): Leverages on-disk zero-RAM paging if True, otherwise loads to cache.
        memmap_path (str, optional): Target binary file location for saving memory maps.
        selected_sensors (List[str]): Exact names of channels to load
            (e.g. ['Tachometer', 'UH Axial Acc', 'UH Radial Acc',
                    'UH Tangential Acc', 'OH Axial Acc', 'OH Radial Acc',
                    'OH Tangential Acc', 'Microphone']).
        target_classes (Union[str, List[str]]): Specific fault types to load, or 'all'.
        target_severities (Union[str, List[str]]): Specific fault intensities to load, or 'all'.
        rpm_range (Union[str, Tuple[float, float]]): Bounding limits filtering specific motor speeds, or 'all'.
        max_workers (int): The number of parallel CPU worker processes allocated, set None to choose all availables.

    Returns:
        Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]: 
            - X: 4D Tensor mapping the signal data blocks.
            - Y: 1D Array mapping string-based fault categories.
            - Meta: Inside a tuple containing (Severity_Array, RPM_Array) respectively.
    """
    loader = MAFAULDARawLoader(
        zarr_path=zarr_path, selected_sensors=selected_sensors, folds=folds,
        labeling_strategy=labeling_strategy, group_misalignment=group_misalignment,
        use_memmap=use_memmap, memmap_path=memmap_path, target_classes=target_classes,
        target_severities=target_severities, rpm_range=rpm_range, max_workers=max_workers,
    )
    return loader.load()


def stratified_file_split(
    y: np.ndarray, 
    train_ratio: float = 0.75, 
    val_ratio: float = 0.25, 
    random_seed: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    A high-level wrapper that automates leakage-free, stratified data splitting 
    and returns ready-to-use NumPy arrays via Fancy Indexing.

    This function wraps `generate_stratified_file_split` to simplify the user pipeline.
    It splits the dataset in a single call while ensuring that:
    1. No physical file spans across different splits (Zero Data Leakage).
    2. The target class distribution remains identical in Train, Val, and Test sets.

    Args:
        y (np.ndarray): The 1D target array containing class labels, shape [N].
        train_ratio (float): Percentage of physical files allocated to Training (default: 0.8).
        val_ratio (float): Percentage of physical files allocated to Validation (default: 0.1).
        random_seed (int): Control seed for shuffling reproducibility (default: 42).

    Returns:
        Tuple[np.array, np.array, np.array]: A nested tuple containing following components:
            1. Indices Splits: train_idx, val_idx, test_idx containing the raw NumPy index arrays.
    """
    # 1. Calling the core stratified file-based index generator
    train_idx, val_idx, test_idx = generate_stratified_file_split(
        y=y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )
    
    return train_idx, val_idx, test_idx

def SlidingWindow(
    X_base: np.ndarray,
    Y_base: np.ndarray,
    window_size: int = 2048,
    step_size: int = 512,
    valid_folds: Optional[List[int]] = None,
    valid_files: Optional[List[int]] = None,
    meta_base: Tuple[np.ndarray, np.ndarray] = (None, None),
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generates structural overlapping sliding window arrays completely inside physical RAM.
    Utilizes optimized C-backend stride views under numpy to maximize throughput.

    Args:
        X_base (np.ndarray): The foundational 4D tensor returned by the primary loader.
        Y_base (np.ndarray): Foundational target string categories.
        window_size (int): Segment sequence length per frame (e.g., 1024).
        step_size (int): Overlap shift increment controlling window stepping density (e.g., 512).
        valid_folds (List[int], optional): Subset lists restricting extraction scopes.
        valid_files (List[int], optional): Subset array of allowed structural file indices to bypass data leakage.
        meta_base (Tuple[np.ndarray, np.ndarray]): Embedded nested metadata tuple (Severity, RPM).

    Returns:
        Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - X_phys: Flattened 3D batch array of dimensions [Windows, Channels, WindowSize].
            - Y_phys: Flattened replicated categorical labels.
            - Meta_phys: Replicated metadata containing (Severity, RPM) matching the slice indices.
    """
    pw = PhysicalSlidingWindow(
        X_base=X_base, Y_base=Y_base, meta_base=meta_base, valid_files=valid_files,
        window_size=window_size, step_size=step_size, valid_folds=valid_folds
    )
    return pw.extract()


def VirtualWindowing(
    X_base: np.ndarray,
    Y_base: np.ndarray,
    window_size: int = 2048,
    step_size: int = 512,
    valid_folds: Optional[List[int]] = None,
    valid_files: Optional[List[int]] = None,
    meta_base: Tuple[np.ndarray, np.ndarray] = (None, None),
) -> VirtualSlidingWindow:
    """
    Spawns a zero-copy Virtual Sliding Window engine over the foundational 4D tensor.
    This allows random access to any window index on-demand with absolute zero memory footprint.

    Args:
        X_base (np.ndarray): The foundational 4D tensor returned by the primary loader.
        Y_base (np.ndarray): Foundational target string categories.
        window_size (int): Segment sequence length per frame (e.g., 1024).
        step_size (int): Overlap shift increment controlling window stepping density (e.g., 512).
        valid_folds (List[int], optional): Subset lists restricting extraction scopes.
        valid_files (List[int], optional): Subset array of allowed structural file indices to bypass data leakage.
        meta_base (Tuple[np.ndarray, np.ndarray]): Embedded nested metadata tuple (Severity, RPM).

    Returns:
        VirtualSlidingWindow: A lightweight virtual engine exposing '.get_window(idx)' and '.total_windows'.
        
    Example:
        >>> vw = mafaulda.VirtualSlidingWindow(X, Y, meta, window_size=1024, step_size=512)
        >>> print(vw.total_windows)
        >>> x_win, y_label, (sev, rpm) = vw.get_window(1000)
    """
    return VirtualSlidingWindow(
        X_base=X_base, Y_base=Y_base, meta_base=meta_base, valid_files=valid_files,
        window_size=window_size, step_size=step_size, valid_folds=valid_folds
    )


def sample_few_shot_tasks(
    X_base: np.ndarray,
    Y_base: np.ndarray,
    numeric_to_string: Dict[int, str],
    window_size: int,
    step_size: int,
    valid_folds: Optional[List[int]] = None,
    valid_files: Optional[List[int]] = None,
    meta_base: Optional[Tuple[np.ndarray, np.ndarray]] = (None, None),
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Samples isolated task frames tailored for Few-Shot / Meta-Learning configurations.
    Enforces strict structural isolation (SRP) and maps indices lazily over disk grids.

    Args:
        X_base (np.ndarray): The core structured 4D input tensor.
        Y_base (np.ndarray): Master string labels array.
        numeric_to_string (Dict[int, str]): Map dictionary translating integers to strings.
                                                e.g., {0: 'normal', 1: 'imbalance', 2: 'misalignment'}
        window_size (int): Segment window size.
        step_size (int): Overlap step increment.
        valid_folds (List[int], optional): Specific folds allocated for task extraction.
        valid_files (List[int], optional): Subset array of allowed structural file indices to bypass data leakage.
        meta_base (Tuple[np.ndarray, np.ndarray]): Bound master metadata tracking metrics.
        seed (int, optional): Pseudo-random generator state value ensuring full task reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - X_task: Collected frame segments matching [Samples, Channels, WindowSize].
            - Y_task: Standardized numeric identifiers mapping to the position within the input tuple.
            - Meta_task: Isolated task metadata segments (Severity, RPM).
    """
    sampler = FewShotSampler(
        X_base=X_base, Y_base=Y_base, numeric_to_string=numeric_to_string,
        window_size=window_size, step_size=step_size, valid_folds=valid_folds,
        meta_base=meta_base, valid_files=valid_files,
    )
    if seed is not None:
        sampler.reset_seed(seed)
    return sampler


def get_pytorch_dataloader(
    X_base: np.ndarray,
    Y_base: np.ndarray,
    window_size: int,
    step_size: int,
    class_to_idx: Dict[str, int],
    batch_size: int = 32,
    shuffle: bool = True,
    valid_folds: Optional[List[int]] = None,
    valid_files: Optional[List[int]] = None,
    **dataloader_kwargs
):
    """
    Assembles a high-performance, native PyTorch DataLoader wrapper around the zero-copy virtual window engine.
    Imports PyTorch lazily to avoid heavy framework initialization overheads.

    Args:
        X_base (np.ndarray): Core structured 4D input tensor.
        Y_base (np.ndarray): Master string labels array.
        window_size (int): Segment window length.
        step_size (int): Overlap shift step increment.
        class_to_idx (Dict[str, int]): Map dictionary translating categorical names into integers.
        batch_size (int): Minibatch density streamed on each iterative step.
        shuffle (bool): Randomizes batch sequence orderings if True.
        valid_folds (List[int], optional): Targeted cross-validation subset regions.
        valid_files (List[int], optional): Subset array of allowed structural file indices to bypass data leakage.
        **dataloader_kwargs: Arbitrary keyword options fed directly to torch.utils.data.DataLoader.

    Returns:
        torch.utils.data.DataLoader: A native, zero-copy loader yielding batches of X, Y.
    """
    from torch.utils.data import DataLoader
    vw = VirtualSlidingWindow(
        X_base=X_base, Y_base=Y_base, valid_files=valid_files,
        window_size=window_size, step_size=step_size, valid_folds=valid_folds
    )
    wrapper = PyTorchMafauldaDataset(virtual_window=vw, class_to_idx=class_to_idx)
    return DataLoader(wrapper.get_dataset(), batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)


def get_tensorflow_dataset(
    X_base: np.ndarray,
    Y_base: np.ndarray,
    window_size: int,
    step_size: int,
    class_to_idx: Dict[str, int],
    batch_size: int = 32,
    valid_folds: Optional[List[int]] = None,
    valid_files: Optional[List[int]] = None,
):
    """
    Builds an optimized tf.data.Dataset pipeline connected directly to the zero-copy sliding matrix.
    Uses standard output signatures to guarantee safe execution across Graph and Eager environments.

    Args:
        X_base (np.ndarray): Core structured 4D input tensor.
        Y_base (np.ndarray): Master string labels array.
        window_size (int): Segment window size.
        step_size (int): Overlap window stepping.
        class_to_idx (Dict[str, int]): Code map dictionary pointing category strings to integer ids.
        batch_size (int): Number of arrays packaged inside each parallel streaming batch.
        valid_folds (List[int], optional): Restrict evaluation or sampling to selected cross-validation spaces.
        valid_files (List[int], optional): Subset array of allowed structural file indices to bypass data leakage.

    Returns:
        tf.data.Dataset: A high-throughput pre-batched and pre-fetched tensorflow stream instance.
    """
    vw = VirtualSlidingWindow(
        X_base=X_base, Y_base=Y_base, valid_files=valid_files,
        window_size=window_size, step_size=step_size, valid_folds=valid_folds
    )
    wrapper = TFMafauldaGenerator(virtual_window=vw, class_to_idx=class_to_idx)
    return wrapper.get_dataset(batch_size=batch_size)


def disk_streamed_concat(
    tensors: Tuple[np.ndarray, ...], 
    axis: int = 0, 
    output_path: str = None, 
    chunk_size: int = 1000
) -> np.memmap:
    """Physically concatenates memory-mapped arrays into a new persistent file.
    
    Streams the data block-by-block using tiny transitional RAM buffers to 
    prevent memory overflow (OOM).

    Args:
        tensors (Tuple[np.ndarray, ...]): List of input memmap arrays.
        output_path (str): File system destination for the new memmap container.
        axis (int): Target axis along which concatenation is executed.
        chunk_size (int): Size of the streaming buffer window along the concatenated axis.

    Returns:
        np.memmap: The newly compiled and mapped contiguous numpy array on disk.
    """
    # 1. Compute and validate structural dimensions
    ndim = tensors[0].ndim
    dtype = tensors[0].dtype
    shapes = [t.shape for t in tensors]
    
    # Calculate unified target shape dimensions
    target_shape = list(shapes[0])
    target_shape[axis] = sum(s[axis] for s in shapes)
    target_shape = tuple(target_shape)
    
    # Secure parent directories
    temp_path = os.path.join(tempfile.gettempdir(), 'mafaulda_ml_ready.dat')
    directory = output_path if output_path else temp_path
    os.makedirs(os.path.dirname(directory), exist_ok=True)
    
    # 2. Allocate the empty physical container directly on the disk
    out_memmap = np.memmap(output_path, dtype=dtype, mode='w+', shape=target_shape)
    
    # 3. Stream data from source files to destination in blocks (Zero-RAM copy)
    current_offset = 0
    for _, src_tensor in enumerate(tensors):
        src_size = src_tensor.shape[axis]
        
        # Iterate and transfer data chunk-by-chunk along the target axis
        for start in range(0, src_size, chunk_size):
            end = min(start + chunk_size, src_size)
            
            # Setup dynamic slicing indices
            src_slices = [slice(None)] * ndim
            src_slices[axis] = slice(start, end)
            
            dest_slices = [slice(None)] * ndim
            dest_slices[axis] = slice(current_offset + start, current_offset + end)
            
            # Read a tiny chunk into RAM and write it instantly back to the new disk file
            out_memmap[tuple(dest_slices)] = src_tensor[tuple(src_slices)]
            
        # Update partition offset for the next file
        current_offset += src_size
        
    # Flush I/O buffers to ensure all bytes are physically written to the hard drive
    out_memmap.flush()
    print(f"✅ Persistent physical concatenation complete: '{output_path}'")
    
    return out_memmap


def feature_extraction_pipeline(
    X_raw: np.ndarray,
    Y_raw: np.ndarray,
    window_size: int,
    step_size: int,
    transform_fn: Callable[[np.ndarray], np.ndarray],
    save_zarr_path: str,
    meta_raw: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
):
    """Orchestrates an end-to-end, Zero-RAM streaming feature extraction pipeline.

    This function serves as a high-level engineering wrapper over the object-oriented 
    ZeroRAMFeatureWorkspace class. It abstracts away class instantiation complexities, 
    dynamically slices sliding windows on the fly, automatically broadcasts 1D file-level 
    metadata into aligned 3D Zarr arrays [Folds, Files, Windows], and processes 
    large datasets file-by-file to prevent out-of-memory (OOM) crashes.

    Args:
        X_draw (np.ndarray): Foundational 4D raw signal tensor. 
                               Shape must be [Folds, Files, Channels, Length].
        Y_raw (np.ndarray): 1D array of original string labels matching the Files dimension.
        window_size (int): Temporal frame length of each sliding window segment.
        step_size (int): Overlap shift increment step size for window sliding.
        transform_fn (Callable): Feature extraction function accepting a 2D window matrix 
                                 of shape [Channels, window_size] and returning a 
                                 transformed NumPy feature vector/matrix.
                               Shape: [Files].
        save_zarr_path (str): File system destination path to compile and secure the Zarr database.
        meta_raw (Tuple): A tuple containing parallel tracking metadata arrays:
                             (Severity array or None, RPM array or None). Shapes: [Files].

    Returns:
        zarr.Group: The direct read-write root group pointer of the persistent Zarr database.
                    Contains chunked 'features', 'labels', 'severity', and 'rpm' arrays.
    """
    # 1. Instantiate the single-responsibility workspace backend internally
    workspace = ZeroRAMFeatureWorkspace(
        window_size=window_size,
        step_size=step_size,
        transform_fn=transform_fn
    )
    
    # 2. Trigger the orchestrated pipeline execution and pass parameters down
    zarr_root = workspace.execute(
        X_raw=X_raw,
        Y_raw=Y_raw,
        meta_raw=meta_raw,
        save_zarr_path=save_zarr_path
    )
    
    return zarr_root


__all__ = [
    "",
    "load_zarr_to_tensor",
    "download",
    "ingest",
    "filter",
    "load",
    "stratified_file_split",
    "SlidingWindow",
    "VirtualWindowing",
    "sample_few_shot_tasks",
    "get_pytorch_dataloader",
    "get_tensorflow_dataset",
    "disk_streamed_concat"
]