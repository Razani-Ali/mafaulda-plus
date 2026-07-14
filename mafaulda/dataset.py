"""
MAFAULDA Core Processing Module

This module handles raw data ingestion, memory-mapped tensor allocation, 
and highly optimized sliding window generation (both physical and virtual).
All classes are designed with maximum memory efficiency and CPU parallelization in mind.
"""

import os
import random
import tempfile
import concurrent.futures
from typing import List, Tuple, Union, Dict, Optional

import zarr
import numpy as np
from tqdm.auto import tqdm


class LabelStrategy:
    """
    Responsibility: Determines how the final string labels and severities 
    are generated based on user-defined configurations.
    """
    def __init__(self, strategy: str, group_misalignment: bool = False):
        """
        Initializes the labeling strategy.
        
        Args:
            strategy (str): The naming strategy (e.g., 'types & severity' or 'only types').
            group_misalignment (bool): If True, groups directional misalignments into a single class.
        """
        self.strategy = strategy
        self.group_misalignment = group_misalignment

    def generate(self, fault_type: str, severity: str) -> Tuple[str, str]:
        """
        Generates the final fault label and severity string.
        
        Args:
            fault_type (str): The raw fault type extracted from metadata.
            severity (str): The raw severity extracted from metadata.
            
        Returns:
            Tuple[str, str]: A tuple containing (final_fault_label, final_severity_label).
        """
        # If the strategy demands concatenation of type and severity
        if self.strategy == 'types & severity':
            return f"{fault_type}{severity}", severity
            
        # If the user wants to group all misalignments (horizontal/vertical) together
        if self.group_misalignment and 'misalignment' in fault_type:
            direction = fault_type.split('-')[0]
            # Return general 'misalignment' class, but append direction to severity
            return 'misalignment', f"{direction}{severity}"
            
        # Default behavior: return exactly as provided
        return fault_type, severity


class MemoryAllocator:
    """
    Responsibility: Manages the allocation of massive multidimensional arrays 
    either in RAM or directly on the physical disk (zero-copy memory mapping).
    """
    @staticmethod
    def create_tensor(shape: tuple, dtype: np.dtype, use_memmap: bool, memmap_path: str = None) -> np.ndarray:
        """
        Allocates a tensor array.
        
        Args:
            shape (tuple): The dimensional shape of the requested tensor.
            dtype (np.dtype): The numerical data type (e.g., np.float16).
            use_memmap (bool): If True, allocates on disk to save RAM.
            memmap_path (str, optional): Target file path for the memory map.
            
        Returns:
            np.ndarray: An allocated array (either in-memory or memory-mapped).
        """
        if use_memmap:
            # Generate a temporary path if no specific path is provided
            if memmap_path is None:
                memmap_path = os.path.join(tempfile.gettempdir(), 'mafaulda_ml_ready.dat')
            # Initialize and return the memory-mapped numpy array in read-write mode
            return np.memmap(memmap_path, dtype=dtype, mode='w+', shape=shape)
            
        # If memmap is disabled, allocate standard RAM-based zeros array
        return np.zeros(shape, dtype=dtype)


class MAFAULDARawLoader:
    """
    Upgraded Loader Engine: Data is structured directly into a Machine Learning 
    ready format right from the start.
    Target Shape: [Folds, Files, Channels, Signal_Length]
    """
    def __init__(self, zarr_path: str,
                 folds: int = 4,
                 labeling_strategy: str = 'only types',
                 group_misalignment: bool = False,
                 use_memmap: bool = True, memmap_path: str = None,
                 selected_sensors: List[str] = 
                    ['Tachometer', 'UH Axial Acc', 'UH Radial Acc',
                    'UH Tangential Acc', 'OH Axial Acc', 'OH Radial Acc',
                    'OH Tangential Acc', 'Microphone'],
                 target_classes: Union[str, List[str]] = 'all',
                 target_severities: Union[str, List[str]] = 'all',
                 rpm_range: Union[str, Tuple[float, float]] = 'all',
                 max_workers: int = None):
        """
        Initializes the raw data loader.
        """
        # Open the Zarr database in read-only mode
        self.db = zarr.open_group(zarr_path, mode='r')
        
        # Extract global sensor mapping to identify column indices dynamically
        global_map = self.db.attrs.get('sensor_mapping', {})
        self.col_indices = [global_map[s] for s in selected_sensors]
        self.num_sensors = len(self.col_indices)
        
        # Ensure folds are at least 1
        self.folds = max(1, folds)
        
        # Instantiate dependencies
        self.labeler = LabelStrategy(labeling_strategy, group_misalignment)
        self.use_memmap = use_memmap
        self.memmap_path = memmap_path
        
        # Store filtering criteria
        self.target_classes = target_classes
        self.target_severities = target_severities
        self.rpm_range = rpm_range

        # Determine thread count (capped at 32 to prevent context-switching overhead)
        self.num_workers = max_workers if max_workers else min(32, (os.cpu_count() or 1) * 4)

    def _get_all_arrays(self, group: zarr.Group) -> List[zarr.Array]:
        """Recursively traverses the Zarr directory tree to find all arrays."""
        arrays = []
        def _search_recursive(current_group):
            for key in current_group.keys():
                item = current_group[key]
                if isinstance(item, zarr.Array):
                    arrays.append(item)
                elif isinstance(item, zarr.Group):
                    _search_recursive(item)
        _search_recursive(group)
        return arrays

    def _filter_arrays(self, all_arrays: List[zarr.Array]) -> Tuple[List[zarr.Array], List[tuple]]:
        """Filters discovered arrays based on user-defined criteria (Class, Severity, RPM)."""
        filtered_arrays = []
        labels_data = []
        
        for arr in all_arrays:
            # Safely extract attributes with fallback defaults
            raw_type = arr.attrs.get('fault_type', 'unknown')
            raw_sev = str(arr.attrs.get('fault_severity', ''))
            rpm_val = float(arr.attrs.get('rpm', 0.0))
            
            # Generate the final naming conventions
            final_y, final_sev = self.labeler.generate(raw_type, raw_sev)
            
            # Apply target class filter
            if self.target_classes != 'all' and final_y not in self.target_classes:
                continue
            # Apply target severity filter
            if self.target_severities != 'all' and final_sev not in self.target_severities:
                continue
            # Apply RPM range boundary filter
            if self.rpm_range != 'all' and not (self.rpm_range[0] <= rpm_val <= self.rpm_range[1]):
                continue
                    
            # If all filters pass, append to the final lists
            filtered_arrays.append(arr)
            labels_data.append((final_y, rpm_val, final_sev))
            
        # Prevent silent failures if no data matches the criteria
        if not filtered_arrays:
            raise ValueError("No files found matching your specified criteria.")
            
        return filtered_arrays, labels_data

    def _allocate_tensors(self, num_files: int, min_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Allocates the primary signal tensor and the parallel metadata arrays."""
        # Calculate the uniform length per fold
        chunk_len = min_length // self.folds
        
        # Define the 4D network-ready shape
        raw_shape = (self.folds, num_files, self.num_sensors, chunk_len)
        
        # Allocate the heavy signal tensor
        X = MemoryAllocator.create_tensor(raw_shape, np.float16, self.use_memmap, self.memmap_path)
        
        # Allocate lightweight metadata arrays in RAM
        Y = np.empty(num_files, dtype=object)
        RPM = np.empty(num_files, dtype=np.float32)
        Severity = np.empty(num_files, dtype=object)
        
        return X, Y, Severity, RPM

    def _populate_tensors(self, X, Y, Severity, RPM, filtered_arrays, labels_data, min_length):
        """Concurrently streams data from disk and writes it into the standardized tensors."""
        chunk_len = min_length // self.folds
        
        def _write_file(idx, zarr_arr, label_tuple):
            # Extract, slice, and reshape the matrix in a single continuous operation
            signal = zarr_arr[:self.folds * chunk_len, self.col_indices]
            
            # Reshape into [Folds, Chunk_Length, Channels]
            signal_folded = signal.reshape(self.folds, chunk_len, self.num_sensors)
            
            # Transpose axes to achieve [Folds, Channels, Chunk_Length] for ML frameworks
            signal_transposed = np.swapaxes(signal_folded, 1, 2)
            
            # Write directly to the pre-allocated memory map or RAM tensor
            X[:, idx, :, :] = signal_transposed
            
            # Assign corresponding metadata parallelly using the identical index
            Y[idx] = label_tuple[0]        # fault_type
            RPM[idx] = label_tuple[1]       # rpm_val
            Severity[idx] = label_tuple[2]   # fault_severity

        # Notify user that a long I/O operation is starting
        print(f"⏳ Starting parallel data population using {self.num_workers} threads...")
        
        # Launch concurrent thread pool to maximize disk throughput
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(_write_file, i, arr, labels_data[i])
                for i, arr in enumerate(filtered_arrays)
            ]
            # Wrap the execution in a progress bar to prevent the app from feeling 'frozen'
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Writing Standardized Tensor"):
                pass
                
        # Flush I/O buffers to ensure all bytes are successfully written to disk
        if self.use_memmap:
            X.flush()

    def load(self) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Main orchestration method: Scans, allocates, populates, and returns the unified arrays.
        """
        print("⏳ Scanning Zarr hierarchy and applying filters...")
        
        # Discover and filter arrays
        all_arrays = self._get_all_arrays(self.db)
        filtered_arrays, labels_data = self._filter_arrays(all_arrays)
        
        # Calculate boundaries
        num_files = len(filtered_arrays)
        min_length = min(arr.shape[0] for arr in filtered_arrays)
        
        # Allocate tensors based on boundaries
        X, Y, Severity, RPM = self._allocate_tensors(num_files, min_length)
        
        # Fill the allocated tensors with actual data
        self._populate_tensors(X, Y, Severity, RPM, filtered_arrays, labels_data, min_length)
        
        print(f"✅ Data Loading Complete! Base Shape: {X.shape}")
        
        # Return standard requested ML output format: X, Y, (Sev, RPM)
        return X, Y, (Severity, RPM)


class VirtualSlidingWindow:
    """
    Fully virtual management of sliding windows across a continuous tensor.
    RAM consumption footprint: 0 bytes.
    """
    def __init__(self, X_base: np.ndarray, Y_base: np.ndarray, 
                 window_size: int = 2048, step_size: int = 512,
                 valid_folds: List[int] = None,
                 valid_files: List[int] = None,
                 meta_base: Tuple[np.ndarray, np.ndarray] = (None, None)):
        """
        Initializes the zero-copy virtual window engine.
        """
        # Store foundational tensor references
        self.X = X_base
        self.Y = Y_base
        self.Severity, self.RPM = meta_base
        
        # Store window mechanics
        self.window_size = window_size
        self.step_size = step_size
        
        # Extract physical dimensions
        self.total_folds, self.total_files, self.channels, self.chunk_len = self.X.shape
        
        # Resolve which folds to virtually iterate over
        self.valid_folds = valid_folds if valid_folds is not None else list(range(self.total_folds))
        self.valid_files = valid_files if valid_files is not None else list(range(self.total_files))
        
        # Calculate mathematical bounds for virtual mapping
        self.windows_per_file = ((self.chunk_len - self.window_size) // self.step_size) + 1
        self.total_windows = len(self.valid_folds) * len(self.valid_files) * self.windows_per_file

    def get_window(self, idx: int) -> Tuple[np.ndarray, str, Tuple[Optional[str], Optional[float]]]:
        """
        Translates a flat conceptual index into a physical multidimensional slice.
        """
        if idx < 0 or idx >= self.total_windows:
            raise IndexError("Index out of bounds")
            
        # Decouple the flat index into specific fold and file locators using integer division
        fold_relative_idx = idx // (len(self.valid_files) * self.windows_per_file)
        remainder = idx % (len(self.valid_files) * self.windows_per_file)
        
        # Decouple remainder to find specific file and internal window index
        file_idx = remainder // self.windows_per_file
        win_idx = remainder % self.windows_per_file
        
        # Map back to physical coordinates
        actual_fold = self.valid_folds[fold_relative_idx]
        actual_file = self.valid_files[file_idx]
        start_pos = win_idx * self.step_size
        end_pos = start_pos + self.window_size
        
        # Perform standard slicing (this yields a fast memory view, not a copy)
        x_win = self.X[actual_fold, actual_file, :, start_pos:end_pos]
        
        # Safely extract metadata if it exists
        sev = None if self.Severity is None else self.Severity[actual_file]
        rpm = None if self.RPM is None else self.RPM[actual_file]

        return x_win, self.Y[actual_file], (sev, rpm)


class PhysicalSlidingWindow:
    """
    Physical extraction of all sliding windows using numpy's C-backend.
    Generates contiguous arrays in RAM without relying on slow Python for-loops.
    """
    def __init__(self, X_base: np.ndarray, Y_base: np.ndarray, 
                 window_size: int, step_size: int,
                 valid_files: List[int] = None, valid_folds: List[int] = None,
                 meta_base: Tuple[np.ndarray, np.ndarray] = (None, None)):
        """
        Initializes the physical extractor engine.
        """
        self.X = X_base
        self.Y = Y_base
        self.Severity, self.RPM = meta_base
        self.window_size = window_size
        self.step_size = step_size
        
        self.total_folds, self.total_files, self.channels, self.chunk_len = self.X.shape
        self.valid_folds = valid_folds if valid_folds is not None else list(range(self.total_folds))
        self.valid_files = valid_files if valid_files is not None else list(range(self.total_files))
        
        self.windows_per_file = ((self.chunk_len - self.window_size) // self.step_size) + 1
        self.total_windows = len(self.valid_folds) * self.total_files * self.windows_per_file

    def extract(self) -> Tuple[np.ndarray, np.ndarray, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Executes the strided extraction and forces a physical copy into RAM.
        """
        print(f"⚠️ Warning: Physically copying {self.total_windows} windows into RAM at C-speed...")
        print("⏳ Please wait. This may take a while depending on your available memory...")
        
        # Filter folds and files in one vectorized operation
        X_valid = self.X[self.valid_folds]
        X_valid = X_valid[:, self.valid_files]
        
        V_folds = len(self.valid_folds)
        V_files = len(self.valid_files)
        
        # Extract memory leap sizes (strides) to manipulate the view mathematically
        stride_fold, stride_file, stride_chan, stride_len = X_valid.strides
        
        # Formulate a 5D illusion of overlapping windows using memory strides (Zero-Copy)
        X_5d = np.lib.stride_tricks.as_strided(
            X_valid,
            shape=(V_folds, V_files, self.windows_per_file, self.channels, self.window_size),
            strides=(stride_fold, stride_file, self.step_size * stride_len, stride_chan, stride_len),
            writeable=False
        )
        
        # Flatten the illusion and force a hard physical copy in RAM (blocking operation)
        X_phys = X_5d.reshape(self.total_windows, self.channels, self.window_size).copy()

        # Filter metadata based on valid_files list before broadcasting
        Y_filtered = self.Y[self.valid_files]
        Y_expanded = np.broadcast_to(Y_filtered.reshape(1, V_files, 1), (V_folds, V_files, self.windows_per_file)).flatten()

        # Safely broadcast metadata if provided
        if self.Severity is not None:
            Sev_filtered = self.Severity[self.valid_files]
            Sev_expanded = np.broadcast_to(Sev_filtered.reshape(1, V_files, 1), (V_folds, V_files, self.windows_per_file)).flatten()
        else:
            Sev_expanded = None

        if self.RPM is not None:
            RPM_filtered = self.RPM[self.valid_files]
            RPM_expanded = np.broadcast_to(RPM_filtered.reshape(1, V_files, 1), (V_folds, V_files, self.windows_per_file)).flatten()
        else:
            RPM_expanded = None
        
        print("✅ Physical window extraction complete.")
        return X_phys, Y_expanded, (Sev_expanded, RPM_expanded)
    

def generate_stratified_file_split(
    y: np.ndarray, 
    train_ratio: float = 0.75, 
    val_ratio: float = 0.25, 
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates stratified random split indices based on file IDs to prevent data leakage.
    
    This function ensures that:
    1. Windows from the same physical file are never split across Train/Val/Test sets.
    2. The class distribution (stratification) is maintained across the splits.

    Args:
        y (np.ndarray): 1D array of labels for each sample.
        train_ratio (float): Proportion of files to include in the train split.
        val_ratio (float): Proportion of files to include in the validation split.
        random_seed (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (train_indices, val_indices, test_indices)
    """

    if (train_ratio + val_ratio) > 1.0:
        raise ValueError(f"Sum of ratios must be less or equal to 1.0, got {train_ratio + val_ratio}")

    np.random.seed(random_seed)
    
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        # 1. Find Related Indices
        label_mask = (y == label)
        class_indices = np.where(label_mask)[0]
        
        # 2. Extract and Shuffle Unique Files
        shuffled_indices = class_indices.copy()
        np.random.shuffle(shuffled_indices)
        
        total_files = len(shuffled_indices)
        if total_files == 0:
            continue
            
        # 3. Slicing calculations for very low file counts (e.g., CWRU/MAFAULDA)
        test_ratio = 1.0 - (train_ratio + val_ratio)
        
        if total_files >= 4 and test_ratio > 0.02 and val_ratio > 0.02:
            # Standard balanced split for adequate file sizes
            train_count = int(np.floor(total_files * train_ratio))
            val_count = int(np.ceil(total_files * val_ratio))
            
            if train_count + val_count >= total_files:
                train_count = max(1, total_files - val_count - 1)
            
            train_end = max(1, train_count)
            val_end = train_end + max(1, val_count)
            
        else:
            # Handle extreme low-file regimes (e.g., exactly 4 files per class)
            if test_ratio <= 0.02 or np.isclose(test_ratio, 0.0):
                # 2-way split (Train / Val only, No Test set)
                if total_files == 4:
                    train_end = 3
                    val_end = 4
                else:
                    train_end = max(1, int(np.round(total_files * train_ratio)))
                    val_end = total_files
            else:
                # 3-way split with very limited files -> Distribution skew is inevitable
                print(
                    f"Class '{label}' has only {total_files} file(s). Performing a strict 3-way split "
                    "in a low-file regime will inevitably skew the statistical class distribution across splits. "
                    "Consider setting val_ratio=0.0 for a clean 2-way split.")
                
                if total_files == 4:
                    train_end = 2  # 2 files for Train
                    val_end = 3    # 1 file for Val (Remaining 1 file goes to Test)
                elif total_files == 3:
                    train_end = 1
                    val_end = 2
                else:
                    train_end = 1
                    val_end = total_files

        # Distribute file nodes directly matching the index matrix
        train_idx.extend(shuffled_indices[:train_end])
        val_idx.extend(shuffled_indices[train_end:val_end])
        test_idx.extend(shuffled_indices[val_end:])
    
    train_idx_arr = np.array(train_idx, dtype=int)
    val_idx_arr = np.array(val_idx, dtype=int)
    test_idx_arr = np.array(test_idx, dtype=int)

    # 6. Final Shuffle
    np.random.shuffle(train_idx_arr)
    np.random.shuffle(val_idx_arr)
    np.random.shuffle(test_idx_arr)
    
    return train_idx_arr, val_idx_arr, test_idx_arr
