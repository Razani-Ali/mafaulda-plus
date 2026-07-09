import os
import zarr
import numpy as np
from zarr.codecs import BloscCodec
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from typing import Dict, Callable, List

# ==========================================
# 1. Zarr Directory Scanner (Zarr v3 Compliant)
# ==========================================
class ZarrScanner:
    """
    Responsibility: Traversing the Zarr directory tree and extracting the paths of all signal arrays.
    Fully compatible with the requirements and new structure of Zarr version 3.
    """
    @staticmethod
    def get_array_paths(group: zarr.Group, current_path: str = "") -> List[str]:
        """
        Recursively scans a Zarr group to find all arrays.

        Args:
            group (zarr.Group): The current Zarr group or root being scanned.
            current_path (str): The accumulated path hierarchy. Defaults to "".

        Returns:
            List[str]: A list containing the string paths of all discovered arrays.
        """
        paths = []
        
        # Extract direct child keys using the native v3 method
        try:
            keys = list(group.keys())
        except AttributeError:
            # Fallback attribute in case the keys method is missing
            keys = list(group)
            
        for key in keys:
            # Build the relative and unique path for each node
            item_path = f"{current_path}/{key}" if current_path else key
            item = group[key]
            
            # In Zarr v3, arrays have a shape attribute
            if hasattr(item, 'shape'):
                paths.append(item_path)
            else:
                # If the node lacks a shape, it is a subgroup (folder), so we recurse into it
                paths.extend(ZarrScanner.get_array_paths(item, item_path))
                
        return paths

# ==========================================
# 2. Signal Processing Logic
# ==========================================
class SignalMatrixProcessor:
    """
    Responsibility: Receiving a raw matrix, applying specified filters, and performing Downsampling.
    """
    def __init__(self, sensor_map: Dict[str, int], filter_map: Dict[str, Callable], downsample_factor: int):
        """
        Initializes the signal processor.

        Args:
            sensor_map (Dict[str, int]): Dictionary mapping sensor names to column indices.
            filter_map (Dict[str, Callable]): Dictionary mapping sensor names to filter functions.
            downsample_factor (int): Integer step-size for decimation/downsampling.
        """
        self.sensor_map = sensor_map
        self.filter_map = filter_map
        self.downsample_factor = downsample_factor

    def process(self, data_matrix: np.ndarray) -> np.ndarray:
        """
        Processes the input matrix column by column based on the provided filters.

        Args:
            data_matrix (np.ndarray): The raw 2D signal array.

        Returns:
            np.ndarray: The filtered and downsampled array.
        """
        processed_matrix = data_matrix.copy()

        for sensor_name, col_idx in self.sensor_map.items():
            raw_signal = processed_matrix[:, col_idx]
            filter_callable = self.filter_map.get(sensor_name)

            if filter_callable is not None:
                processed_matrix[:, col_idx] = filter_callable(raw_signal)

        if self.downsample_factor > 1:
            processed_matrix = processed_matrix[::self.downsample_factor, :]

        return processed_matrix

# ==========================================
# 3. IO Orchestrator (The Engine - Zarr v3 Confirmed)
# ==========================================
class ParallelZarrFilterEngine:
    """
    Responsibility: Orchestrating reading from the source file, passing it to the 
    math processor, and writing to the destination Zarr.
    """
    def __init__(self, src_store_path: str, dst_store_path: str, 
                 filter_map: Dict[str, Callable], max_workers: int = 8, 
                 downsample_factor: int = 1):
        """
        Initializes the concurrent IO execution engine.

        Args:
            src_store_path (str): Path to the source Zarr store.
            dst_store_path (str): Path to the destination Zarr store.
            filter_map (Dict[str, Callable]): Dictionary of filter executables.
            max_workers (int, optional): Thread/process boundary. Defaults to 8.
            downsample_factor (int, optional): Step size for signal length reduction. Defaults to 1.
        """
        self.src_store_path = src_store_path
        self.dst_store_path = dst_store_path
        self.max_workers = max_workers
        
        self.src_root = zarr.open_group(src_store_path, mode='r')
        self.dst_root = zarr.open_group(dst_store_path, mode='a')
        
        # 1:1 copy of global metadata
        self.dst_root.attrs.update(self.src_root.attrs)
        
        if 'sensor_mapping' not in self.src_root.attrs:
            raise KeyError("❌ 'sensor_mapping' not found in source Zarr global attributes.")
        
        sensor_map = self.src_root.attrs['sensor_mapping']
        self.processor = SignalMatrixProcessor(sensor_map, filter_map, downsample_factor)
        
        # Using official v3 compression settings
        self.compressor = BloscCodec(cname='lz4', clevel=5, shuffle='shuffle')

    def _process_io_task(self, path: str):
        """
        Internal worker task handling the complete IO cycle for a single array.

        Args:
            path (str): The hierarchical internal Zarr path.

        Returns:
            str: The processed path string, used for tracking completion.
        """
        src_array = self.src_root[path]
        raw_matrix = src_array[:]
        
        processed_matrix = self.processor.process(raw_matrix)
        
        # Reverting to standard v3 parameters (compressors as a list) to remove warnings
        dst_array = self.dst_root.require_array(
            name=path,
            shape=processed_matrix.shape,
            chunks=src_array.chunks,
            dtype=processed_matrix.dtype,
            compressors=[self.compressor]
        )
        dst_array[:] = processed_matrix
        dst_array.attrs.update(src_array.attrs) # 1:1 copy of local array metadata
        
        return path

    def run(self):
        """Execute the main workflow pipeline"""
        array_paths = ZarrScanner.get_array_paths(self.src_root)
        total_files = len(array_paths)
        print(f"🚀 Found {total_files} arrays. Starting parallel filtering and downsampling...")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_io_task, path): path for path in array_paths}
            
            with tqdm(total=total_files, desc="Processing Data") as bar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"\n❌ Error processing path {futures[future]}: {e}")
                    finally:
                        bar.update(1)
        
        print(f"✅ Pipeline complete. Data saved to: {self.dst_store_path}")