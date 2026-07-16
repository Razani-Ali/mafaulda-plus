"""
Module: transform_zarr.py
Description: An advanced dimension-agnostic feature extraction framework powered by Zarr.
Implements Zero-RAM streaming ingestion and Flat-Map indexing to eliminate memory limits
and prevent Data Leakage via File-Level strict validation.
"""

import os
import gc
import zarr
import numpy as np
from tqdm.auto import tqdm
from typing import Callable, Tuple, Any, List, Optional


class ZeroRAMFeatureWorkspace:
    """
    Manages offline feature extraction by streaming chunks into a Zarr database.
    Prevents RAM overflow (OOM) by executing file-by-file and flushing instantly.
    """

    @staticmethod
    def transform_and_save(
        X_domain: np.ndarray,
        Y_domain: np.ndarray,
        meta_domain: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
        transform_fn: Callable[[np.ndarray], np.ndarray],
        save_zarr_path: str
    ) -> zarr.Group:
        """
        Executes a Zero-RAM feature extraction pipeline.

        Args:
            X_domain (np.ndarray): 5D Input Tensor [Folds, Files, Windows, Channels, Length].
            Y_domain (np.ndarray): 1D array of string labels matching the Files dimension.
            meta_domain (Tuple): Aligned (Severity, RPM) arrays matching the Files dimension.
            transform_fn (Callable): Feature extraction function accepting [Channels, Length].
            save_zarr_path (str): Destination disk path for the Zarr database.
            
        Returns:
            zarr.Group: The root group of the newly created Zarr database.
        """
        # 1. Extract Structural Dimensions
        if X_domain.ndim != 5:
            raise ValueError(f"❌ Expected 5D tensor [F, N, W, C, L], got {X_domain.ndim}D.")
            
        F, N, W, C, L = X_domain.shape
        Severity, RPM = meta_domain

        print(f"📦 Detected 5D Tensor: {F} Folds, {N} Files, {W} Windows per file.")

        # 2. Dry Run: Discover Feature Dimensions automatically
        print("🧪 Performing Dry Run to discover feature boundaries...")
        sample_window = X_domain[0, 0, 0, :, :]  # Shape: [Channels, Length]
        sample_feature = transform_fn(sample_window)
        D = sample_feature.shape  # E.g., (128,) or (10, 10)
        
        feature_shape = (F, N, W) + D
        print(f"📏 Discovered Feature Shape: {D} -> Target Tensor: {feature_shape}")

        # 3. Initialize Lazy Zarr Database Structure (No RAM Allocation)
        os.makedirs(os.path.dirname(save_zarr_path) if os.path.dirname(save_zarr_path) else ".", exist_ok=True)
        store = zarr.DirectoryStore(save_zarr_path)
        root = zarr.group(store=store, overwrite=True)

        # Create physical disk arrays (Chunked specifically per file to optimize DataLoader reads)
        z_feat = root.zeros('features', shape=feature_shape, chunks=(1, 1, W) + D, dtype=sample_feature.dtype)
        
        # Save parallel metadata identically mapped to the Files dimension
        root.array('labels', Y_domain, dtype=object)
        if Severity is not None:
            root.array('severity', Severity, dtype=object)
        if RPM is not None:
            root.array('rpm', RPM, dtype=np.float32)

        # 4. Streaming Execution (The Zero-RAM Engine)
        # We loop over FILES to load only tiny fractions of data at a time
        for file_idx in tqdm(range(N), desc="Extracting Features (Zero-RAM)", unit="file"):
            
            # Load ALL Folds and Windows for THIS SPECIFIC FILE into RAM
            # Shape: [Folds, Windows, Channels, Length]
            file_chunk = X_domain[:, file_idx, :, :, :]
            
            # Flatten conceptual dimensions for uniform iteration [F*W, C, L]
            file_chunk_flat = file_chunk.reshape(F * W, C, L)
            
            extracted_batch = []
            for i in range(F * W):
                extracted_batch.append(transform_fn(file_chunk_flat[i]))
                
            # Restructure back to the physical multidimensional format [Folds, Windows, D]
            extracted_np = np.stack(extracted_batch, axis=0)
            extracted_reshaped = extracted_np.reshape((F, W) + D)
            
            # Flush strictly into the disk location for this file
            z_feat[:, file_idx, ...] = extracted_reshaped
            
            # 🧹 AGGRESSIVE RAM CLEANUP: Destroy temporary objects and run GC
            del file_chunk, file_chunk_flat, extracted_batch, extracted_np, extracted_reshaped
            gc.collect()

        print(f"✅ Extraction complete! Persistent Database secured at: '{save_zarr_path}'")
        return root


class FlatMapFeatureDataset:
    """
    High-Performance $O(1)$ Dataloader structure designed to read pre-extracted 
    Zarr features using linear logical indexing, governed strictly by File-Level Validations.
    """
    
    def __init__(self, zarr_path: str, 
                 valid_folds: List[int], 
                 valid_files: List[int]):
        """
        Initializes the Flat-Map Reader.
        
        Args:
            zarr_path (str): Path to the compiled Zarr database.
            valid_folds (List[int]): Array of allowed Folds (prevents Covariate Shift).
            valid_files (List[int]): Array of allowed Files (prevents Data Leakage).
        """
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"❌ Zarr database not found at {zarr_path}")

        # Connect to the persistent disk database in Read-Only mode
        self.root = zarr.open_group(zarr_path, mode='r')
        
        # Link lazy pointers
        self.features = self.root['features']
        self.labels = self.root['labels']
        self.severity = self.root['severity'] if 'severity' in self.root else None
        self.rpm = self.root['rpm'] if 'rpm' in self.root else None
        
        # Enforce Anti-Leakage constraints
        self.valid_folds = valid_folds
        self.valid_files = valid_files
        self.windows_per_file = self.features.shape[2]
        
        # The total logical length available to the model training loop
        self.total_samples = len(self.valid_folds) * len(self.valid_files) * self.windows_per_file

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str, Tuple[Optional[str], Optional[float]]]:
        """
        Maps a 1D batch request into a 3D physical coordinate seamlessly.
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("❌ Index out of bounds")

        # 1. Decode Flat Index into Relative Pointers
        fold_relative_idx = idx // (len(self.valid_files) * self.windows_per_file)
        remainder = idx % (len(self.valid_files) * self.windows_per_file)

        file_relative_idx = remainder // self.windows_per_file
        win_idx = remainder % self.windows_per_file

        # 2. Map Relative Pointers to Actual Database Addresses
        actual_fold = self.valid_folds[fold_relative_idx]
        actual_file = self.valid_files[file_relative_idx]

        # 3. Retrieve Data Lazily from Disk (Extremely fast due to smart chunking)
        x_feature = self.features[actual_fold, actual_file, win_idx]
        y_label = self.labels[actual_file]
        
        # Safely fetch metadata
        sev_val = self.severity[actual_file] if self.severity is not None else None
        rpm_val = self.rpm[actual_file] if self.rpm is not None else None

        # Return standardized output tuple expected by PyTorch/TensorFlow Wrappers
        return x_feature, y_label, (sev_val, rpm_val)
    
    def load_all_to_ram(self) -> Tuple[np.ndarray, np.ndarray, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """Loads and materializes the entire filtered dataset partition into system RAM.
        
        This is highly useful for downstream tasks such as classical Machine Learning 
        (e.g., SVM, Random Forest), quick visualization (t-SNE/UMAP), or rapid prototyping.
        It uses Zarr's fast orthogonal indexing and broadcasts 1D file-level metadata 
        to perfectly align with the flattened window-level samples.

        Returns:
            X_ram (np.ndarray): Contiguous features array of shape (total_samples, *D).
            Y_ram (np.ndarray): Aligned 1D array of labels of shape (total_samples,).
            meta_ram (Tuple): Aligned (Severity, RPM) arrays of shape (total_samples,).
        """
        print(f"⚡ Materializing {self.total_samples} samples into system RAM...")
        
        # 1. Extract and flatten the multi-dimensional features using Zarr's orthogonal selection
        # Shape output before reshape: (len(valid_folds), len(valid_files), windows_per_file, *D)
        features_subset = self.features.get_orthogonal_selection((
            self.valid_folds, 
            self.valid_files, 
            slice(None)
        ))
        
        # Flatten the temporal/file dimensions to match the 1D virtual sample count
        feature_dims = self.features.shape[3:]  # Captures the extracted feature dimensions D
        X_ram = features_subset.reshape((self.total_samples,) + feature_dims)
        
        # 2. Extract and align 1D File-level Labels to match the 3D window structure
        # We slice labels first, then broadcast them across Folds (dim 0) and Windows (dim 2)
        labels_subset = self.labels[self.valid_files]
        
        # Shape transition: (N_valid,) -> (1, N_valid, 1) -> (F_valid, N_valid, W) -> (total_samples,)
        Y_ram = np.broadcast_to(
            labels_subset[None, :, None], 
            (len(self.valid_folds), len(self.valid_files), self.windows_per_file)
        ).flatten()
        
        # 3. Safely broadcast and align parallel metadata trackers if they exist
        Sev_ram = None
        if self.severity is not None:
            sev_subset = self.severity[self.valid_files]
            Sev_ram = np.broadcast_to(
                sev_subset[None, :, None], 
                (len(self.valid_folds), len(self.valid_files), self.windows_per_file)
            ).flatten()
            
        RPM_ram = None
        if self.rpm is not None:
            rpm_subset = self.rpm[self.valid_files]
            RPM_ram = np.broadcast_to(
                rpm_subset[None, :, None], 
                (len(self.valid_folds), len(self.valid_files), self.windows_per_file)
            ).flatten()
            
        print(f"🏆 Materialization Complete! X_ram Shape: {X_ram.shape}, Y_ram Shape: {Y_ram.shape}")
        return X_ram, Y_ram, (Sev_ram, RPM_ram)