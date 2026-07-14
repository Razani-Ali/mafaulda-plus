"""
MAFAULDA Frameworks Module
==========================
Specialized interfaces for PyTorch and TensorFlow, strictly adhering to the 
core engine's structural output.
Final batch output format: X, Y, (Severity, RPM)
"""

from typing import Dict

class PyTorchMafauldaDataset:
    """
    Native wrapper for the PyTorch DataLoader.
    Preserves the standardized core output structure: X, Y, (Severity, RPM).
    """
    def __init__(self, virtual_window, class_to_idx: Dict[str, int]):
        # Lazy import to prevent framework conflicts and overhead
        import torch
        from torch.utils.data import Dataset
        
        self.torch = torch
        self.Dataset = Dataset
        self.vw = virtual_window
        self.class_to_idx = class_to_idx
        
        class _InnerDataset(self.Dataset):
            def __init__(self, parent):
                self.parent = parent
                
            def __len__(self):
                # Return the total available windows calculated by the virtual engine
                return self.parent.vw.total_windows
                
            def __getitem__(self, idx):
                # Retrieve the standard output from the central core engine
                x_np, y_str, _ = self.parent.vw.get_window(idx)
                
                # Convert the NumPy arrays/values into native PyTorch tensors
                x_tensor = self.parent.torch.tensor(x_np, dtype=self.parent.torch.float32)
                y_tensor = self.parent.torch.tensor(self.parent.class_to_idx[y_str], dtype=self.parent.torch.long)
                # Convert to tensor ONLY if the value is not None, otherwise preserve None

                # Return the exact requested nested format to the DataLoader
                return x_tensor, y_tensor
                
        self.dataset = _InnerDataset(self)

    def get_dataset(self):
        """Returns the instantiated PyTorch Dataset."""
        return self.dataset


class TFMafauldaGenerator:
    """
    Native generator for TensorFlow tf.data.Dataset.
    Utilizes modern output_signature while preserving the structured metadata format.
    """
    def __init__(self, virtual_window, class_to_idx: Dict[str, int]):
        # Lazy import to prevent framework conflicts and overhead
        import tensorflow as tf
        self.tf = tf
        self.vw = virtual_window
        self.class_to_idx = class_to_idx

    def _generator(self):
        """Internal generator yielding individual window instances."""
        for i in range(self.vw.total_windows):
            x_np, y_str, _ = self.vw.get_window(i)
            # Yield the structured format aligning identically with the core output
            yield x_np, self.class_to_idx[y_str]

    def get_dataset(self, batch_size: int = 32):
        """
        Constructs the high-performance tf.data.Dataset pipeline.
        
        Args:
            batch_size (int): Number of samples per batch. Defaults to 32.
            
        Returns:
            tf.data.Dataset: A batched and prefetched TensorFlow dataset.
        """
        # Define the output signature precisely as a nested, encapsulated structure
        dataset = self.tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                self.tf.TensorSpec(shape=(self.vw.channels, self.vw.window_size), dtype=self.tf.float16), # X
                self.tf.TensorSpec(shape=(), dtype=self.tf.int32),                                       # Y
                (
                    self.tf.TensorSpec(shape=(), dtype=self.tf.string),                                  # Severity
                    self.tf.TensorSpec(shape=(), dtype=self.tf.float32)                                  # RPM
                )
            )
        )
        # Apply batching and parallel prefetching for maximum GPU throughput
        return dataset.batch(batch_size).prefetch(self.tf.data.AUTOTUNE)