import os
import gc
import zarr
from zarr.storage import LocalStore
import numpy as np
from tqdm.auto import tqdm
from typing import Callable, Tuple, Optional, Any, Optional, List, Union, Dict
import random


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

        try:
            root.create_array(
                name=name,
                data=broadcasted_view,
                chunks=shape
            )

        except (AttributeError, TypeError, ValueError):
            try:
                z_arr = root.create_array(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    chunks=shape,
                    fill_value="" if dtype == object else 0
                )
                z_arr[:] = broadcasted_view

            except (AttributeError, TypeError):
                try:
                    z_arr = root.zeros(name, shape=shape, dtype=dtype, chunks=shape)
                    z_arr[:] = broadcasted_view
                except AttributeError:
                    root[name] = broadcasted_view

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
        save_zarr_path: str,
        sensor_names: Optional[List[str]] = None
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
        
        print(f"📏 Feature Dimension Discovered: {D} -> Mapping X data with Shape: {feature_shape}")

        # Initialize Storage Group
        os.makedirs(os.path.dirname(save_zarr_path) if os.path.dirname(save_zarr_path) else ".", exist_ok=True)
        try:
            store = LocalStore(root=save_zarr_path)
            root = zarr.open_group(store=store, mode='w')
        except (ImportError, AttributeError):
            try:
                store = zarr.DirectoryStore(save_zarr_path)
                root = zarr.group(store=store, overwrite=True)
            except AttributeError:
                root = zarr.open_group(save_zarr_path, mode='w')

        # Allocate features tensor on disk
        try:
            z_feat = root.create_array(
                name='features', 
                shape=feature_shape, 
                chunks=((1, 1, W) + D), 
                dtype=dummy_feature.dtype, 
                fill_value=0
            )
        except TypeError:
            z_feat = root.zeros(
                'features', 
                shape=feature_shape, 
                chunks=((1, 1, W) + D), 
                dtype=dummy_feature.dtype
            )

        # 🚀 Broadcast metadata to match [F, N, W] structural layout instantly
        self._broadcast_and_save_meta(root, 'labels', Y_raw, meta_shape, object)
        if Severity is not None: 
            self._broadcast_and_save_meta(root, 'severity', Severity, meta_shape, object)
        if RPM is not None: 
            self._broadcast_and_save_meta(root, 'rpm', RPM, meta_shape, np.float32)
        if sensor_names is not None:
            root.attrs['sensor_mapping'] = sensor_names

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
    


def _get_valid_file_indices(
    labels_3d: np.ndarray,
    severities_3d: Optional[np.ndarray],
    rpms_3d: Optional[np.ndarray],
    target_classes: Union[str, List[str]],
    target_severities: Union[str, List[str]],
    rpm_range: Union[str, Tuple[float, float]]
) -> List[int]:
    """Evaluates filtering conditions and returns authorized file indices."""
    # Since metadata is broadcasted to [F, N, W], we extract the 1D file-level base using [0, :, 0]
    base_labels = labels_3d[0, :, 0]
    base_sevs = severities_3d[0, :, 0] if severities_3d is not None else None
    base_rpms = rpms_3d[0, :, 0] if rpms_3d is not None else None
    
    N = len(base_labels)
    valid_indices = []

    for i in range(N):
        # 1. Evaluate Class
        if target_classes != 'all':
            allowed_classes = [target_classes] if isinstance(target_classes, str) else target_classes
            if base_labels[i] not in allowed_classes:
                continue
                
        # 2. Evaluate Severity
        if target_severities != 'all' and base_sevs is not None:
            allowed_sevs = [target_severities] if isinstance(target_severities, str) else target_severities
            if base_sevs[i] not in allowed_sevs:
                continue
                
        # 3. Evaluate RPM Range
        if rpm_range != 'all' and base_rpms is not None:
            min_rpm, max_rpm = rpm_range
            if not (min_rpm <= base_rpms[i] <= max_rpm):
                continue
                
        valid_indices.append(i)
        
    return valid_indices

def _get_valid_channel_indices(
    saved_sensors: List[str], 
    selected_sensors: Union[str, List[str]]
) -> List[int]:
    """Maps requested sensor names to their exact numerical channel indices."""
    if selected_sensors == 'all':
        return list(range(len(saved_sensors)))
        
    requested = [selected_sensors] if isinstance(selected_sensors, str) else selected_sensors
    valid_indices = []
    
    for sensor in requested:
        if sensor in saved_sensors:
            valid_indices.append(saved_sensors.index(sensor))
        else:
            print(f"⚠️ Warning: Requested sensor '{sensor}' was not found in the Zarr database.")
            
    return valid_indices


def load_zarr_to_tensor(
    zarr_path: str, 
    as_memmap: bool = True, 
    memmap_path: Optional[str] = None,
    selected_sensors: Union[str, List[str]] = 'all',
    target_classes: Union[str, List[str]] = 'all',
    target_severities: Union[str, List[str]] = 'all',
    rpm_range: Union[str, Tuple[float, float]] = 'all'
) -> Tuple[np.ndarray, np.ndarray, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """Reads, filters, and materializes a target-specific subset from the Zarr database.

    Args:
        zarr_path: Path to the source Zarr feature database.
        as_memmap: If True, streams the filtered subset into a new memory-map file.
        memmap_path: Destination for the memory-map (required if as_memmap=True).
        selected_sensors: Names of target channels to load, or 'all'.
        target_classes: Specific fault types to load, or 'all'.
        target_severities: Specific fault intensities to load, or 'all'.
        rpm_range: Bounding limits (min, max) for motor speeds, or 'all'.
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"❌ Zarr storage not found at: {zarr_path}")

    print(f"🔍 Opening Zarr database at '{zarr_path}'...")
    root = zarr.open_group(zarr_path, mode='r')
    z_feat = root['features']
    
    # Extract complete broadcasted metadata blocks directly into RAM
    full_Y = root['labels'][:]
    full_Sev = root['severity'][:] if 'severity' in root else None
    full_RPM = root['rpm'][:] if 'rpm' in root else None

    # --- APPLY FILE-LEVEL FILTERS ---
    valid_files = _get_valid_file_indices(full_Y, full_Sev, full_RPM, target_classes, target_severities, rpm_range)
    if not valid_files:
        raise ValueError("❌ No files matched the given filtering criteria.")
    print(f"🎯 File Filter: {len(valid_files)} out of {z_feat.shape[1]} files selected.")

    # Slice the RAM-loaded metadata to match only valid files
    # Preserves shape [Folds, N_valid, Windows]
    Y_ram = full_Y[:, valid_files, :]
    Sev_ram = full_Sev[:, valid_files, :] if full_Sev is not None else None
    RPM_ram = full_RPM[:, valid_files, :] if full_RPM is not None else None

    # --- APPLY CHANNEL-LEVEL FILTERS ---
    valid_channels = slice(None)
    new_feat_shape = list(z_feat.shape)
    new_feat_shape[1] = len(valid_files)  # Update N dimension

    if selected_sensors != 'all':
        saved_sensors = root.attrs.get('sensor_mapping', [])
        if not saved_sensors:
            print("⚠️ Warning: No sensor mapping found in Zarr. Channel filtering skipped.")
        else:
            valid_channels = _get_valid_channel_indices(saved_sensors, selected_sensors)
            if not valid_channels:
                raise ValueError("❌ No valid sensors matched the selection.")
            print(f"🎯 Channel Filter: Loading {len(valid_channels)} out of {len(saved_sensors)} channels.")
            # Assuming channel is the 4th dimension: [F, N, W, Channels, ...]
            new_feat_shape[3] = len(valid_channels)

    new_feat_shape = tuple(new_feat_shape)

    # --- MATERIALIZATION (IN-RAM vs MEMMAP) ---
    if not as_memmap:
        print(f"⚡ Loading filtered subset directly into RAM...")
        # Get orthogonal selection via NumPy advanced indexing directly from Zarr
        # z_feat[:, valid_files, :, valid_channels, ...]
        if selected_sensors != 'all':
            X_ram = z_feat.get_orthogonal_selection((slice(None), valid_files, slice(None), valid_channels))
        else:
            X_ram = z_feat.get_orthogonal_selection((slice(None), valid_files, slice(None)))
            
        print("🏆 Full In-RAM loading complete successfully.")
        return X_ram, Y_ram, (Sev_ram, RPM_ram)

    # Streaming Zero-RAM approach to new Memmap
    if memmap_path is None:
        raise ValueError("❌ 'memmap_path' must be specified when 'as_memmap' is set to True.")

    print(f"💾 Streaming filtered subset into Memory-Map at '{memmap_path}'...")
    os.makedirs(os.path.dirname(memmap_path) if os.path.dirname(memmap_path) else ".", exist_ok=True)
    X_memmap = np.memmap(memmap_path, dtype=z_feat.dtype, mode='w+', shape=new_feat_shape)
    
    # Stream gracefully file-by-file
    for new_idx, original_file_idx in enumerate(tqdm(valid_files, desc="Streaming Filtered X", unit="file")):
        # Extract the specific file chunk
        temp_buffer = z_feat[:, original_file_idx, ...]
        
        # Apply channel filter if requested
        if selected_sensors != 'all':
            temp_buffer = temp_buffer[:, :, valid_channels, ...]
            
        # Write to the contiguous memmap location
        X_memmap[:, new_idx, ...] = temp_buffer
        X_memmap.flush()

    print(f"✅ Filtered Memmap conversion complete. Shape: {X_memmap.shape}")
    return X_memmap, Y_ram, (Sev_ram, RPM_ram)



class VirtualFeatureWindow:
    """Provides O(1) flat-mapped access to multi-dimensional feature tensors.
    
    This intermediary class translates linear indices (from PyTorch/TensorFlow) 
    into absolute physical coordinates within a 4D+ feature tensor: 
    [Folds, Files, Windows, *Feature_Dims]. It natively enforces data isolation 
    (Anti-Leakage) via the authorized fold and file-level constraints.
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        meta_base: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
        valid_folds: List[int],
        valid_files: List[int]
    ):
        """Initializes the virtual mapping boundaries and links the tensors.
        
        Args:
            X (np.ndarray): The processed features tensor [Folds, Files, Windows, *Dims].
            Y (np.ndarray): The broadcasted labels tensor [Folds, Files, Windows].
            meta_base (Tuple): Parallel tracking arrays (Severity, RPM), matching the shape.
            valid_folds (List[int]): Authorized temporal fold indices for this exact split.
            valid_files (List[int]): Authorized physical file indices for this exact split.
        """
        # Bind references to the foundational arrays (In-RAM or Memory-Mapped)
        self.X = X
        self.Y = Y
        self.Severity, self.RPM = meta_base
        
        # Lock in the allowed physical boundaries to prevent Covariate Shift and Data Leakage
        self.valid_folds = valid_folds
        self.valid_files = valid_files
        
        # Dynamically extract the structural window counts from the 3rd dimension of the tensor
        self.windows_per_file = self.X.shape[2]
        
        # Calculate the absolute total of flat logical samples exposed to the deep learning frameworks
        self.total_windows = len(self.valid_folds) * len(self.valid_files) * self.windows_per_file

    def __len__(self) -> int:
        """Returns the total number of valid samples available in this partition."""
        return self.total_windows

    def get_window(self, idx: int) -> Tuple[np.ndarray, Any, Tuple[Any, Any]]:
        """Retrieves a single feature sample and its aligned metadata via O(1) translation.
        
        This method strictly complies with the `PyTorchMafauldaDataset` and 
        `TFMafauldaGenerator` interfaces, returning the precise requested tuple.
        
        Args:
            idx (int): The flat logical index requested by the Dataloader (0 to total_windows - 1).
            
        Returns:
            Tuple: A strictly formatted tuple containing (x_feature, y_label, (severity, rpm)).
        """
        # Enforce strict boundary checks to prevent memory overflow errors
        if idx < 0 or idx >= self.total_windows:
            raise IndexError("❌ Logical Dataloader Index is out of bounds!")
            
        # 1. Decode the sequential 1D index into relative spatial pointers
        fold_relative_idx = idx // (len(self.valid_files) * self.windows_per_file)
        remainder = idx % (len(self.valid_files) * self.windows_per_file)
        
        file_relative_idx = remainder // self.windows_per_file
        win_idx = remainder % self.windows_per_file
        
        # 2. Route the relative pointers to the exact authorized physical coordinates
        actual_fold = self.valid_folds[fold_relative_idx]
        actual_file = self.valid_files[file_relative_idx]
        
        # 3. Execute zero-copy extraction directly from the bound tensors
        x_feature = self.X[actual_fold, actual_file, win_idx]
        y_label = self.Y[actual_fold, actual_file, win_idx]
        
        # 4. Safely extract metadata indices only if they were supplied dynamically
        sev_val = self.Severity[actual_fold, actual_file, win_idx] if self.Severity is not None else None
        rpm_val = self.RPM[actual_fold, actual_file, win_idx] if self.RPM is not None else None
        
        # Return the exact standard output signature requested by the deep learning frameworks
        return x_feature, y_label, (sev_val, rpm_val)



class FewShotSampler:
    """A high-performance, dimension-agnostic sampler for Few-Shot and Meta-Learning.
    
    Adapted for the new Zero-RAM feature extraction pipeline, this sampler expects 
    pre-windowed tensors of shape [Folds, Files, Windows, *Feature_Dims] and 
    broadcasted metadata of shape [Folds, Files, Windows]. It directly indexes 
    the data without executing any temporal slicing.

    Attributes:
        X (np.ndarray): Extracted features tensor [Folds, Files, Windows, *Dims].
        Y (np.ndarray): Broadcasted string labels tensor [Folds, Files, Windows].
        Severity (np.ndarray, optional): Broadcasted severity tensor.
        RPM (np.ndarray, optional): Broadcasted RPM tensor.
        numeric_to_string (Dict[int, str]): Map of integer IDs to string names.
        total_folds (int): Total temporal folds.
        total_files (int): Total physical files.
        windows_per_file (int): Pre-calculated windows per file.
    """
    
    def __init__(self, X_base: np.ndarray, Y_base: np.ndarray, 
                 numeric_to_string: Dict[int, str],
                 valid_folds: Optional[List[int]] = None, 
                 valid_files: Optional[List[int]] = None,
                 meta_base: Optional[Tuple[np.ndarray, np.ndarray]] = (None, None),
                 seed: Optional[int] = None):
        """Initializes the sampler with pre-windowed multidimensional data."""
        self.X = X_base
        self.Y = Y_base
        self.numeric_to_string = numeric_to_string
        self.Severity, self.RPM = meta_base
        
        # Extract direct physical dimensions from the pre-windowed tensor
        self.total_folds = self.X.shape[0]
        self.total_files = self.X.shape[1]
        self.windows_per_file = self.X.shape[2]
        
        self.valid_folds = valid_folds if valid_folds is not None else list(range(self.total_folds))
        self.valid_files = valid_files if valid_files is not None else list(range(self.total_files))
        
        self.file_indices_by_class = self._map_files_to_classes()
        self.label_frequencies = self._calculate_label_frequencies()
        
        # Enforce deterministic behavior if a fixed random seed is explicitly provided
        if seed is not None:
            random.seed(seed)

    def _map_files_to_classes(self) -> dict:
        """Extracts class categories from the broadcasted 3D labels tensor."""
        mapping = {}
        allowed_set = set(self.valid_files)
        
        # Since labels are broadcasted [F, N, W], extracting from [0, :, 0] yields 
        # the 1D base array of file-level categorical strings safely.
        file_level_labels = self.Y[0, :, 0]
        
        for i, label in enumerate(file_level_labels):
            if i in allowed_set:
                if label not in mapping:
                    mapping[label] = []
                mapping[label].append(i)
        return mapping

    def _calculate_label_frequencies(self) -> dict:
        return {
            label: len(indices) * len(self.valid_folds) * self.windows_per_file
            for label, indices in self.file_indices_by_class.items()
        }

    def _validate_inputs(self, target_numeric_classes: Tuple[int, ...], samples_per_class: Tuple[int, ...]):
        if len(target_numeric_classes) != len(samples_per_class):
            raise ValueError("❌ Shape Error: Length of target classes must match samples per class!")

        for num_label, required_samples in zip(target_numeric_classes, samples_per_class):
            string_name = self.numeric_to_string.get(num_label)
            if string_name not in self.file_indices_by_class:
                raise ValueError(f"❌ Class '{string_name}' not found in active split!")
            if self.label_frequencies[string_name] < required_samples:
                raise ValueError(f"❌ Capacity Error: Class '{string_name}' has insufficient windows.")

    def _translate_flat_to_coordinates(self, flat_idx: int, available_files: List[int]) -> Tuple[int, int, int]:
        fold_relative_idx = flat_idx // (len(available_files) * self.windows_per_file)
        remainder = flat_idx % (len(available_files) * self.windows_per_file)

        file_relative_idx = remainder // self.windows_per_file
        win_idx = remainder % self.windows_per_file

        actual_fold = self.valid_folds[fold_relative_idx]
        actual_file = available_files[file_relative_idx]
        return actual_fold, actual_file, win_idx

    def _sample_single_class(self, num_label: int, required_samples: int) -> Tuple[list, list, list, list]:
        sampled_x, sampled_y, sampled_sev, sampled_rpm = [], [], [], []
        string_name = self.numeric_to_string[num_label]
        available_files = self.file_indices_by_class[string_name]
        
        chosen_flat_indices = random.sample(range(self.label_frequencies[string_name]), required_samples)

        for flat_idx in chosen_flat_indices:
            fold, actual_file, win_idx = self._translate_flat_to_coordinates(flat_idx, available_files)
            
            # Direct indexing natively mapping to the pre-windowed Zarr feature matrix
            sampled_x.append(self.X[fold, actual_file, win_idx])
            sampled_y.append(num_label)
            
            sampled_sev.append(self.Severity[fold, actual_file, win_idx] if self.Severity is not None else None)
            sampled_rpm.append(self.RPM[fold, actual_file, win_idx] if self.RPM is not None else None)

        return sampled_x, sampled_y, sampled_sev, sampled_rpm

    def _post_process_and_shuffle(self, s_x: list, s_y: list, s_sev: list, s_rpm: list) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        combined = list(zip(s_x, s_y, s_sev, s_rpm))
        random.shuffle(combined)

        X_final = np.array([item[0] for item in combined])
        Y_final = np.array([item[1] for item in combined], dtype=np.int64)
        
        Sev_final = np.array([item[2] for item in combined], dtype=object) if self.Severity is not None else None
        RPM_final = np.array([item[3] for item in combined], dtype=np.float32) if self.RPM is not None else None
        
        return X_final, Y_final, (Sev_final, RPM_final)

    def sample(self, target_numeric_classes: Tuple[int, ...], 
               samples_per_class: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        self._validate_inputs(target_numeric_classes, samples_per_class)
        all_sampled_x, all_sampled_y, all_sampled_sev, all_sampled_rpm = [], [], [], []

        for num_label, required_samples in zip(target_numeric_classes, samples_per_class):
            x_c, y_c, sev_c, rpm_c = self._sample_single_class(num_label, required_samples)
            all_sampled_x.extend(x_c)
            all_sampled_y.extend(y_c)
            all_sampled_sev.extend(sev_c)
            all_sampled_rpm.extend(rpm_c)

        return self._post_process_and_shuffle(all_sampled_x, all_sampled_y, all_sampled_sev, all_sampled_rpm)