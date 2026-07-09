import random
import numpy as np
from typing import List, Tuple, Dict, Optional

class FewShotSampler:
    """
    A high-performance, strategic sampler designed for Few-Shot and Meta-Learning tasks.
    Enforces strict Single Responsibility Principle (SRP) and layout isolation.
    
    It decouples the ML algorithm from underlying string labels by mapping requested 
    numerical class indices (e.g., 0, 1, 2) directly to their database string names,
    returning clean, single-integer target labels ideal for CrossEntropyLoss.
    """
    def __init__(self, X_base: np.ndarray, Y_base: np.ndarray, 
                 numeric_to_string: Dict[int, str],
                 window_size: int = 2048, step_size: int = 512,
                 valid_folds: Optional[List[int]] = None, 
                 meta_base: Optional[Tuple[np.ndarray, np.ndarray]] = (None, None)):
        """
        Initializes the Few-Shot Sampler.

        Args:
            X_base (np.ndarray): Foundational 4D memory-mapped or In-RAM tensor.
            Y_base (np.ndarray): Master 1D array of original string labels.
            numeric_to_string (Dict[int, str]): Map dictionary translating integers to strings.
                                                e.g., {0: 'normal', 1: 'imbalance', 2: 'misalignment'}
            window_size (int): Segment sequence frame length. Defaults to 2048.
            step_size (int): Overlap shift increment step size. Defaults to 512.
            valid_folds (List[int], optional): Subset lists restricting extraction scopes.
            meta_base (Tuple[np.ndarray, np.ndarray], optional): Bound master metadata (Severity, RPM).
                                                                Defaults to (None, None).
        """
        self.X = X_base
        self.Y = Y_base
        self.numeric_to_string = numeric_to_string
        self.window_size = window_size
        self.step_size = step_size
        
        self.Severity, self.RPM = meta_base
        
        self.total_folds, self.total_files, self.channels, self.chunk_len = self.X.shape
        self.valid_folds = valid_folds if valid_folds is not None else list(range(self.total_folds))
        self.windows_per_file = ((self.chunk_len - self.window_size) // self.step_size) + 1
        
        # Extract physical mappings and class capacities
        self.file_indices_by_class = self._map_files_to_classes()
        self.label_frequencies = self._calculate_label_frequencies()

    def _map_files_to_classes(self) -> dict:
        """Responsibility: Maps underlying binary file positions based on string categories."""
        mapping = {}
        for i, label in enumerate(self.Y):
            if label not in mapping:
                mapping[label] = []
            mapping[label].append(i)
        return mapping

    def _calculate_label_frequencies(self) -> dict:
        """Responsibility: Calculates exact virtual windows available per category."""
        return {
            label: len(indices) * len(self.valid_folds) * self.windows_per_file
            for label, indices in self.file_indices_by_class.items()
        }

    def reset_seed(self, seed: int):
        """Responsibility: Controls the pseudo-random seed generator state."""
        random.seed(seed)

    def _validate_inputs(self, target_numeric_classes: Tuple[int, ...], samples_per_class: Tuple[int, ...]):
        """Responsibility: Structural validation of requested numerical categories and database bounds."""
        if len(target_numeric_classes) != len(samples_per_class):
            raise ValueError("❌ Shape Error: The length of target numeric classes must match samples per class!")

        for num_label, required_samples in zip(target_numeric_classes, samples_per_class):
            if num_label not in self.numeric_to_string:
                raise ValueError(f"❌ Key Error: Numerical class ID '{num_label}' is missing from the injected map!")
            
            string_name = self.numeric_to_string[num_label]
            if string_name not in self.file_indices_by_class:
                raise ValueError(f"❌ Database Error: String class '{string_name}' was not found during ingestion!")
                
            if self.label_frequencies[string_name] < required_samples:
                raise ValueError(
                    f"❌ Capacity Error: Class '{string_name}' has {self.label_frequencies[string_name]} windows, "
                    f"which is less than the requested {required_samples} samples!"
                )

    def _translate_flat_to_coordinates(self, flat_idx: int, available_files: List[int]) -> Tuple[int, int, int, int]:
        """Responsibility: Mathematical mapping of virtual linear indices into disk coordinates."""
        fold_relative_idx = flat_idx // (len(available_files) * self.windows_per_file)
        remainder = flat_idx % (len(available_files) * self.windows_per_file)

        file_relative_idx = remainder // self.windows_per_file
        win_idx = remainder % self.windows_per_file

        actual_fold = self.valid_folds[fold_relative_idx]
        actual_file = available_files[file_relative_idx]
        start_pos = win_idx * self.step_size
        end_pos = start_pos + self.window_size

        return actual_fold, actual_file, start_pos, end_pos

    def _sample_single_class(self, num_label: int, required_samples: int) -> Tuple[list, list, list, list]:
        """Responsibility: Random selection of window coordinates and data stream assembly for one class ID."""
        sampled_x, sampled_y, sampled_sev, sampled_rpm = [], [], [], []
        
        string_name = self.numeric_to_string[num_label]
        available_files = self.file_indices_by_class[string_name]
        chosen_flat_indices = random.sample(range(self.label_frequencies[string_name]), required_samples)

        for flat_idx in chosen_flat_indices:
            fold, file_idx, start, end = self._translate_flat_to_coordinates(flat_idx, available_files)
            
            sampled_x.append(self.X[fold, file_idx, :, start:end])
            sampled_y.append(num_label) # Store the integer class ID directly (e.g., 0 or 2)
            
            # Safely capture metadata if it exists
            if self.Severity is not None:
                sampled_sev.append(self.Severity[file_idx])
            if self.RPM is not None:
                sampled_rpm.append(self.RPM[file_idx])

        return sampled_x, sampled_y, sampled_sev, sampled_rpm

    def _post_process_and_shuffle(self, s_x: list, s_y: list, s_sev: list, s_rpm: list) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Responsibility: Unified integration, random shuffling, and formatting to production arrays."""
        combined = list(zip(s_x, s_y, s_sev, s_rpm))
        random.shuffle(combined)

        X_final = np.array([item[0] for item in combined])
        Y_final = np.array([item[1] for item in combined], dtype=np.int64) # 1D array of integer labels
        Sev_final = np.array([item[2] for item in combined], dtype=object) if self.Severity is not None else None
        RPM_final = np.array([item[3] for item in combined], dtype=np.float32) if self.RPM is not None else None
        
        return X_final, Y_final, (Sev_final, RPM_final)

    def sample(self, target_numeric_classes: Tuple[int, ...], 
               samples_per_class: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Orchestrates the entire execution pipeline. Accepts numeric IDs and drops integer array labels.
        
        Args:
            target_numeric_classes: Tuple of integer IDs representing the chosen classes (e.g., (0, 2))
            samples_per_class: Parallel tuple specifying required window slices per ID (e.g., (50, 5))
            
        Returns:
            X_final: Matched 3D task array [Samples, Channels, WindowSize]
            Y_final: 1D NumPy int64 array containing explicit class index integers [Samples,]
            Meta_final: Packed metadata trackers matching the extracted instances (Severity, RPM)
        """
        self._validate_inputs(target_numeric_classes, samples_per_class)

        all_sampled_x, all_sampled_y, all_sampled_sev, all_sampled_rpm = [], [], [], []

        # Iterate over each requested class with a visual progress bar
        for num_label, required_samples in zip(target_numeric_classes, samples_per_class):
            x_c, y_c, sev_c, rpm_c = self._sample_single_class(num_label, required_samples)
            all_sampled_x.extend(x_c)
            all_sampled_y.extend(y_c)
            all_sampled_sev.extend(sev_c)
            all_sampled_rpm.extend(rpm_c)

        results = self._post_process_and_shuffle(all_sampled_x, all_sampled_y, all_sampled_sev, all_sampled_rpm)
        
        return results