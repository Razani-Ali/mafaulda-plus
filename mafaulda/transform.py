import os
import gc
import zarr
import numpy as np
from tqdm.auto import tqdm
from typing import Callable, Tuple, Optional, Any, Optional


class ZeroRAMFeatureWorkspace:
    """A highly modular, Zero-RAM feature extraction pipeline obeying the Single Responsibility Principle (SRP).
    
    This upgraded version broadcasts 1D file-level metadata into 3D configurations [Folds, Files, Windows]
    seamlessly inside Zarr for immediate downstream multi-framework ingestion without data leakage.
    """

    def __init__(self, window_size: int, step_size: int,
                 transform_fn: Callable[[np.ndarray], np.ndarray]):
        self.window_size = window_size
        self.step_size = step_size
        self.transform_fn = transform_fn

    def _calculate_window_count(self, signal_length: int) -> int:
        """Responsibility: Mathematical projection of the sliding window boundaries."""
        return ((signal_length - self.window_size) // self.step_size) + 1

    def _broadcast_and_save_meta(self, root: zarr.Group, name: str,
                                 data: np.ndarray,
                                 shape: Tuple[int, int, int],
                                 dtype: Any):
        
        """Responsibility: Vectorized broadcasting of 1D file metadata into 3D space directly on disk."""
        # Shape shift: (N,) -> (1, N, 1) -> (F, N, W)
        broadcasted_view = np.broadcast_to(data[None, :, None], shape)
        root.array(name, broadcasted_view, dtype=dtype)

    def _process_single_file(self, file_tensor: np.ndarray,
                             num_windows: int, feature_dims: Tuple[int, ...]
                             ) -> np.ndarray:
        
        """Responsibility: Generating temporal windows and executing features extraction for one file."""
        F, C, L = file_tensor.shape
        file_features_buffer = np.empty((F, num_windows) + feature_dims, dtype=np.float32)

        for f in range(F):
            for w in range(num_windows):
                start_idx = w * self.step_size
                end_idx = start_idx + self.window_size
                
                # Zero-copy dynamic window slicing
                temporal_window = file_tensor[f, :, start_idx:end_idx]
                file_features_buffer[f, w] = self.transform_fn(temporal_window)

        return file_features_buffer

    def execute(
        self, 
        X_raw: np.ndarray, 
        Y_raw: np.ndarray, 
        meta_raw: Tuple[Optional[np.ndarray], Optional[np.ndarray]], 
        save_zarr_path: str
    ) -> zarr.Group:
        """Responsibility: Orchestrating the end-to-end pipeline and enforcing Broadcast rules."""
        if X_raw.ndim != 4:
            raise ValueError(f"❌ Expected 4D tensor [Folds, Files, Channels, Length], got {X_raw.ndim}D.")
            
        F, N, C, L = X_raw.shape
        Severity, RPM = meta_raw
        
        # Geometrical calculations
        W = self._calculate_window_count(L)
        meta_shape = (F, N, W)  # Target 3D shape for all tracking indicators

        # Dry Run for dynamic dimension discovery
        dummy_window = np.zeros((C, self.window_size), dtype=X_raw.dtype)
        dummy_feature = self.transform_fn(dummy_window)
        D = dummy_feature.shape
        feature_shape = (F, N, W) + D
        
        print(f"📏 Feature Dimension Discovered: {D} -> Mapping 3D Metadata with Shape: {meta_shape}")

        # Initialize Storage Group
        os.makedirs(os.path.dirname(save_zarr_path) if os.path.dirname(save_zarr_path) else ".", exist_ok=True)
        store = zarr.DirectoryStore(save_zarr_path)
        root = zarr.group(store=store, overwrite=True)

        # Allocate features tensor on disk
        z_feat = root.zeros('features', shape=feature_shape, chunks=(1, 1, W) + D, dtype=dummy_feature.dtype)
        
        # 🚀 Broadcast metadata to match [F, N, W] structural layout instantly
        self._broadcast_and_save_meta(root, 'labels', Y_raw, meta_shape, object)
        if Severity is not None: 
            self._broadcast_and_save_meta(root, 'severity', Severity, meta_shape, object)
        if RPM is not None: 
            self._broadcast_and_save_meta(root, 'rpm', RPM, meta_shape, np.float32)

        # Streaming Core Loop
        for file_idx in tqdm(range(N), desc="Streaming Feature Extraction", unit="file"):
            file_chunk = X_raw[:, file_idx, :, :]
            extracted_block = self._process_single_file(file_chunk, W, D)
            
            # Flush block to disk
            z_feat[:, file_idx, ...] = extracted_block
            
            del file_chunk, extracted_block
            gc.collect()

        print(f"✅ Secure Zarr DB Compiled at: '{save_zarr_path}'")
        return root
    

def load_zarr_to_tensor(
    zarr_path: str, 
    as_memmap: bool = True, 
    memmap_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """Reads a fully broadcasted Zarr database and materializes the dataset.
    
    To optimize hardware resources, it allows memory-mapping the heavy features 
    tensor (X) to the disk while loading the lightweight metadata arrays (Y, Severity, RPM) 
    directly into system RAM.

    Args:
        zarr_path (str): Path to the source Zarr feature database directory.
        as_memmap (bool): If True, streams the heavy feature matrix into an out-of-core 
                          np.memmap file to save RAM. If False, loads it entirely into RAM.
        memmap_path (str, optional): Destination disk path for the features memory map 
                                     (e.g., 'features.dat'). Required if as_memmap is True.

    Returns:
        X (np.ndarray): Processed feature tensor, either as np.memmap or In-RAM array. 
                        Shape: [Folds, Files, Windows, *Feature_Dims]
        Y (np.ndarray): 1D/3D array of broadcasted string labels fully loaded in RAM.
        Meta (Tuple): Aligned tracking arrays loaded in RAM: (Severity, RPM).
    """
    # 1. Establish read-only connection to the persistent Zarr storage
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"❌ Zarr storage not found at: {zarr_path}")

    print(f"🔍 Connecting to Zarr database at '{zarr_path}'...")
    root = zarr.open_group(zarr_path, mode='r')
    z_feat = root['features']
    
    # 2. Extract lightweight vectors directly into system RAM (Negligible footprint)
    print("🧠 Ingesting lightweight metadata arrays directly into RAM...")
    Y_ram = root['labels'][:]
    Sev_ram = root['severity'][:] if 'severity' in root else None
    RPM_ram = root['rpm'][:] if 'rpm' in root else None

    # 3. Handle Full In-RAM Materialization if requested
    if not as_memmap:
        print(f"⚡ Loading heavy feature tensor of shape {z_feat.shape} entirely into RAM...")
        X_ram = z_feat[:]
        print("🏆 Full In-RAM loading complete successfully.")
        return X_ram, Y_ram, (Sev_ram, RPM_ram)

    # 4. Handle Zero-RAM Memory-Mapped Path for X only
    if memmap_path is None:
        raise ValueError("❌ 'memmap_path' must be specified when 'as_memmap' is set to True.")

    print(f"💾 Constructing physical Memory-Map container for X at '{memmap_path}'...")
    # Ensure the destination directory exists on the host machine
    os.makedirs(os.path.dirname(memmap_path) if os.path.dirname(memmap_path) else ".", exist_ok=True)
    
    # Allocate a blank contiguous physical file on the hard drive matching X dimensions
    X_memmap = np.memmap(memmap_path, dtype=z_feat.dtype, mode='w+', shape=z_feat.shape)
    
    # Extract structural partition bounds to execute streaming
    F, N, W = z_feat.shape[:3]
    
    # Stream heavy data file-by-file to completely isolate memory growth
    for file_idx in tqdm(range(N), desc="Streaming X to Memmap", unit="file"):
        # Fetch one single file block from Zarr into temporary execution buffer
        temp_buffer = z_feat[:, file_idx, ...]
        
        # Write the buffer immediately into the targeted disk block of the memmap
        X_memmap[:, file_idx, ...] = temp_buffer
        
        # Flush operating system buffers to enforce immediate disk writes
        X_memmap.flush()
        
    print(f"✅ Conversion complete. X is mapped to disk. Shape: {X_memmap.shape}")
    return X_memmap, Y_ram, (Sev_ram, RPM_ram)