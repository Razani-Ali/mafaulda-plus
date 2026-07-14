import random
import numpy as np
from typing import List, Tuple, Dict, Optional

class FewShotSampler:
    """A high-performance, memory-efficient sampler for Few-Shot and Meta-Learning.
    
    This sampler enforces layout isolation and prevents data leakage by selecting 
    samples strictly based on partitioned physical files (File-Level Splitting). 
    It decouples string labels from ML models by mapping numerical class indices 
    directly to their corresponding string identifiers.

    Attributes:
        X (np.ndarray): Foundational multi-dimensional tensor (e.g., 5D shape).
        Y (np.ndarray): Master 1D array of original string labels for files.
        numeric_to_string (Dict[int, str]): Mapping of integer class IDs to string names.
        window_size (int): Segment sequence frame length for sliding windows.
        step_size (int): Overlap shift increment step size.
        Severity (np.ndarray, optional): Array of raw fault severity labels.
        RPM (np.ndarray, optional): Array of raw shaft speeds.
        total_folds (int): Total folds available in the input tensor.
        total_files (int): Total files available in the input tensor.
        valid_folds (List[int]): Folds authorized for task sampling.
        valid_files (List[int]): Physical file indices authorized for task sampling (prevents leakage).
        windows_per_file (int): Pre-calculated windows available in each file.
        file_indices_by_class (Dict[str, List[int]]): Mapping of class names to allowed file indices.
        label_frequencies (Dict[str, int]): Total virtual window capacity available per class.
    """
    
    def __init__(self, X_base: np.ndarray, Y_base: np.ndarray, 
                 numeric_to_string: Dict[int, str],
                 window_size: int = 2048, step_size: int = 512,
                 valid_folds: Optional[List[int]] = None, 
                 valid_files: Optional[List[int]] = None,
                 meta_base: Optional[Tuple[np.ndarray, np.ndarray]] = (None, None)):
        """Initializes the Few-Shot Sampler with structured file partitions.

        Args:
            X_base (np.ndarray): Foundational 5D memory-mapped or In-RAM tensor.
            Y_base (np.ndarray): Master 1D array of original string labels.
            numeric_to_string (Dict[int, str]): Mapping of integer labels to string classes.
            window_size (int): Temporal frame length. Defaults to 2048.
            step_size (int): Shift step for slicing. Defaults to 512.
            valid_folds (List[int], optional): Subset of folds to sample from.
            valid_files (List[int], optional): Subset of allowed file indices (e.g., train/val/test splits).
            meta_base (Tuple[np.ndarray, np.ndarray], optional): Metadata arrays (Severity, RPM).
        """
        # Bind core signal and label array references
        self.X = X_base
        self.Y = Y_base
        self.numeric_to_string = numeric_to_string
        self.window_size = window_size
        self.step_size = step_size
        
        # Unpack structural metadata trackers (Severity and RPM arrays)
        self.Severity, self.RPM = meta_base
        
        # Extract base physical dimensions of the machinery engine tensor
        self.total_folds = self.X.shape[0]
        self.total_files = self.X.shape[1]
        
        # Configure default execution boundaries if specific lists are omitted
        self.valid_folds = valid_folds if valid_folds is not None else list(range(self.total_folds))
        self.valid_files = valid_files if valid_files is not None else list(range(self.total_files))
        
        # Calculate dynamic window boundaries based on input dimensionality
        if self.X.ndim == 5:
            # 5D Layout already pre-windowed: [Folds, Files, Channels, Windows, Length]
            self.windows_per_file = self.X.shape[3]
        else:
            # 4D Layout requiring mathematical step calculations: [Folds, Files, Channels, Signal_Length]
            self.windows_per_file = ((self.X.shape[-1] - self.window_size) // self.step_size) + 1
        
        # Parse active database nodes and extract computational capacities
        self.file_indices_by_class = self._map_files_to_classes()
        self.label_frequencies = self._calculate_label_frequencies()

    def _map_files_to_classes(self) -> dict:
        """Maps physical file IDs to their respective string categories.

        Returns:
            dict: Dictionary mapping string class names to lists of authorized file indices.
        """
        mapping = {}
        # Convert valid files to a set for O(1) membership lookup speed
        allowed_set = set(self.valid_files)
        
        # Isolate and group files matching string targets while enforcing split boundaries
        for i, label in enumerate(self.Y):
            if i in allowed_set:
                if label not in mapping:
                    mapping[label] = []
                mapping[label].append(i)
        return mapping

    def _calculate_label_frequencies(self) -> dict:
        """Calculates total virtual window capacities per machinery category.

        Returns:
            dict: Dictionary containing maximum sample limits for each active class.
        """
        # Total capacity = (number of files) * (number of active folds) * (windows per file)
        return {
            label: len(indices) * len(self.valid_folds) * self.windows_per_file
            for label, indices in self.file_indices_by_class.items()
        }

    def reset_seed(self, seed: int):
        """Enforces strict reproducibility for deterministic task generation.

        Args:
            seed (int): Integer pseudo-random seed state value.
        """
        random.seed(seed)

    def _validate_inputs(self, target_numeric_classes: Tuple[int, ...], samples_per_class: Tuple[int, ...]):
        """Performs validation checks against requested shapes and database bounds.

        Args:
            target_numeric_classes (Tuple[int, ...]): Requested class IDs for the episode.
            samples_per_class (Tuple[int, ...]): Number of shots per requested class.

        Raises:
            ValueError: If there is a mismatch in input lengths, invalid class IDs, or capacity limits.
        """
        # Ensure classes and shots have a 1:1 mapping
        if len(target_numeric_classes) != len(samples_per_class):
            raise ValueError("❌ Shape Error: The length of target numeric classes must match samples per class!")

        # Verify capacity constraints for each requested class inside the active file split
        for num_label, required_samples in zip(target_numeric_classes, samples_per_class):
            if num_label not in self.numeric_to_string:
                raise ValueError(f"❌ Key Error: Numerical class ID '{num_label}' is missing from the injected map!")
            
            string_name = self.numeric_to_string[num_label]
            if string_name not in self.file_indices_by_class:
                raise ValueError(f"❌ Database Error: String class '{string_name}' was not found in active split!")
                
            if self.label_frequencies[string_name] < required_samples:
                raise ValueError(
                    f"❌ Capacity Error: Class '{string_name}' has only {self.label_frequencies[string_name]} windows, "
                    f"but {required_samples} samples were requested!"
                )

    def _translate_flat_to_coordinates(self, flat_idx: int, available_files: List[int]) -> Tuple[int, int, int]:
        """Translates a virtual linear index to explicit multidimensional tensor coordinates.

        Args:
            flat_idx (int): A sequential index ranging from 0 to class capacity.
            available_files (List[int]): Allowed physical file indices for this specific class.

        Returns:
            Tuple[int, int, int]: Calculated coordinates (actual_fold, actual_file, win_idx).
        """
        # Isolate the fold coordinate using sequential integer division
        fold_relative_idx = flat_idx // (len(available_files) * self.windows_per_file)
        remainder = flat_idx % (len(available_files) * self.windows_per_file)

        # Isolate the file coordinate relative to authorized splits
        file_relative_idx = remainder // self.windows_per_file
        # Isolate the starting window offset
        win_idx = remainder % self.windows_per_file

        # Map relative pointers to exact physical storage addresses
        actual_fold = self.valid_folds[fold_relative_idx]
        actual_file = available_files[file_relative_idx]

        return actual_fold, actual_file, win_idx

    def _sample_single_class(self, num_label: int, required_samples: int) -> Tuple[list, list, list, list]:
        """Extracts random sliding window slices and metadata for a single target category.

        Args:
            num_label (int): Numeric target class ID.
            required_samples (int): Number of shots to extract.

        Returns:
            Tuple[list, list, list, list]: Lists containing (sampled_signals, labels, severities, rpms).
        """
        sampled_x, sampled_y, sampled_sev, sampled_rpm = [], [], [], []
        
        # Fetch underlying string name and map associated files for the class
        string_name = self.numeric_to_string[num_label]
        available_files = self.file_indices_by_class[string_name]
        
        # Sample non-overlapping flat indices randomly without replacement
        chosen_flat_indices = random.sample(range(self.label_frequencies[string_name]), required_samples)

        # Process each sampled index and extract zero-copy data slices
        for flat_idx in chosen_flat_indices:
            # Map flat index to explicit physical dimensions
            fold, actual_file, win_idx = self._translate_flat_to_coordinates(flat_idx, available_files)
            
            # Retrieve segment slices safely according to input dimensional shape
            if self.X.ndim == 5:
                # Direct indexing for pre-windowed 5D layouts: [Folds, Files, Channels, Windows, Length]
                x_slice = self.X[fold, actual_file, :, win_idx, :]
            else:
                # Dynamic sliding window slicing for standard 4D tensors
                start_pos = win_idx * self.step_size
                end_pos = start_pos + self.window_size
                x_slice = self.X[fold, actual_file, :, start_pos:end_pos]
                
            sampled_x.append(x_slice)
            sampled_y.append(num_label)
            
            # Map and extract aligned physics metadata trackers using the true file index
            if self.Severity is not None:
                sampled_sev.append(self.Severity[actual_file])
            if self.RPM is not None:
                sampled_rpm.append(self.RPM[actual_file])

        return sampled_x, sampled_y, sampled_sev, sampled_rpm

    def _post_process_and_shuffle(self, s_x: list, s_y: list, s_sev: list, s_rpm: list) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Consolidates collected data batches, executes shuffling, and formats outputs.

        Args:
            s_x (list): List of extracted signal slices.
            s_y (list): List of numerical labels.
            s_sev (list): List of severity trackers.
            s_rpm (list): List of RPM speeds.

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]: Standard production tensors.
        """
        # Pack elements into a single list of tuples to maintain alignment during shuffling
        combined = list(zip(s_x, s_y, s_sev, s_rpm))
        random.shuffle(combined)

        # Reconstruct into optimized contiguous NumPy arrays
        X_final = np.array([item[0] for item in combined])
        Y_final = np.array([item[1] for item in combined], dtype=np.int64)
        
        # Safely convert tracking arrays only if they were initialized in the database
        Sev_final = np.array([item[2] for item in combined], dtype=object) if self.Severity is not None else None
        RPM_final = np.array([item[3] for item in combined], dtype=np.float32) if self.RPM is not None else None
        
        return X_final, Y_final, (Sev_final, RPM_final)

    def sample(self, target_numeric_classes: Tuple[int, ...], 
               samples_per_class: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generates an N-Way K-Shot task/episode for meta-learning.

        Args:
            target_numeric_classes (Tuple[int, ...]): Tuple of class IDs to include (e.g., (0, 1) for 2-Way).
            samples_per_class (Tuple[int, ...]): Shots per class (e.g., (5, 5) for 5-Shot).

        Returns:
            X_final: High-performance 3D signal tensor [Samples, Channels, WindowSize]
            Y_final: 1D NumPy int64 array containing explicit class index integers [Samples,]
            Meta_final: Aligned metadata tuple containing (Severity, RPM) trackers
        """
        # Ensure split capacities and targets are mathematically valid
        self._validate_inputs(target_numeric_classes, samples_per_class)

        all_sampled_x, all_sampled_y, all_sampled_sev, all_sampled_rpm = [], [], [], []

        # Iterate and extract samples for each class sequentially
        for num_label, required_samples in zip(target_numeric_classes, samples_per_class):
            x_c, y_c, sev_c, rpm_c = self._sample_single_class(num_label, required_samples)
            all_sampled_x.extend(x_c)
            all_sampled_y.extend(y_c)
            all_sampled_sev.extend(sev_c)
            all_sampled_rpm.extend(rpm_c)

        # Consolidate, shuffle across ways, and return production tensors
        results = self._post_process_and_shuffle(all_sampled_x, all_sampled_y, all_sampled_sev, all_sampled_rpm)
        
        return results