"""
MAFAULDA Ingestion Module

This module is responsible for scanning raw CSV datasets, extracting domain-specific 
metadata from folder structures, reading heavy matrices via C-engines, and writing 
them into a highly optimized, compressed Zarr (v3) binary database using parallel processing.
"""

import os
import pandas as pd
import numpy as np
import zarr
from zarr.codecs import BloscCodec
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

# ==========================================
# 1. Metadata Management
# ==========================================

class MAFAULDAMetadata:
    """
    Responsibility: Extracting, formatting, and managing auxiliary metadata for each file.
    """
    
    SENSOR_MAP = {
        'Tachometer': 0, 'UH Axial Acc': 1, 'UH Radial Acc': 2,
        'UH Tangential Acc': 3, 'OH Axial Acc': 4, 'OH Radial Acc': 5,
        'OH Tangential Acc': 6, 'Microphone': 7
    }

    @staticmethod
    def extract_from_path(file_path: str, data_dir: str) -> dict:
        """
        Takes a file path and extracts categorical metadata (fault type, severity, and RPM).

        Args:
            file_path (str): The absolute or relative path to the individual CSV file.
            data_dir (str): The root directory of the dataset to calculate relative paths.

        Returns:
            dict: A dictionary containing the sanitized Zarr array path and its attributes.
        """
        path_obj = Path(file_path)
        rel_path = path_obj.relative_to(data_dir)
        parts = rel_path.parts
        rel_str = str(rel_path)

        fault_type, fault_severity = "unknown", ""

        # Determine fault type based on the directory path structure (User Logic)
        if "normal" in parts:
            fault_type, fault_severity = "normal", ""
        elif "imbalance" in parts:
            idx = parts.index("imbalance")
            fault_type, fault_severity = "imbalance", parts[idx + 1]
        elif "horizontal-misalignment" in rel_str:
            idx = parts.index("horizontal-misalignment")
            fault_type, fault_severity = "horizontal-misalignment", parts[idx + 1]
        elif "vertical-misalignment" in rel_str:
            idx = parts.index("vertical-misalignment")
            fault_type, fault_severity = "vertical-misalignment", parts[idx + 1]
        elif "overhang" in parts:
            idx = parts.index("overhang")
            fault_type = f"overhang-{parts[idx + 1]}"
            fault_severity = parts[idx + 2]
        elif "underhang" in parts:
            idx = parts.index("underhang")
            fault_type = f"underhang-{parts[idx + 1]}"
            fault_severity = parts[idx + 2]

        # Calculate nominal RPM based on the filename
        try:
            filename_val = float(path_obj.stem)  # Extract filename without the .csv extension
            rpm = round(filename_val * 60)
        except ValueError:
            rpm = -1  # Fallback to -1 if the filename is not a valid numerical float

        # Generate a unique and safe string path for Zarr storage (removing file extension)
        zarr_array_path = str(rel_path.with_suffix('')).replace('\\', '/')

        return {
            "zarr_path": zarr_array_path,
            "attributes": {
                "fault_type": fault_type,
                "fault_severity": fault_severity,
                "rpm": rpm,
                "original_filename": path_obj.name
            }
        }

# ==========================================
# 2. Data Reading
# ==========================================

class CSVReader:
    """
    Responsibility: Reading heavy text files and converting them to NumPy matrices.
    """
    
    @staticmethod
    def read(file_path: str, dtype: np.dtype = np.float16) -> np.ndarray:
        """
        Reads a raw CSV file rapidly and casts it into an optimized NumPy array.

        Args:
            file_path (str): The path to the CSV file.
            dtype (np.dtype, optional): The target numerical precision. Defaults to np.float16.

        Returns:
            np.ndarray: The parsed multi-dimensional signal array.
        """
        # Use pandas with the C engine for maximum speed when parsing 250k rows
        # Assumes the CSV files do not contain headers.
        df = pd.read_csv(file_path, header=None, engine='c')
        
        # Cast to target dtype to significantly reduce the memory footprint
        return df.values.astype(dtype)

# ==========================================
# 3. Zarr Storage Manager (Zarr v3 Compatible)
# ==========================================

class ZarrDatabase:
    """
    Responsibility: Managing the compressed binary file, chunking, and storing metadata.
    """
    
    def __init__(self, store_path: str):
        """
        Initializes the Zarr binary store configuration.

        Args:
            store_path (str): The physical path where the Zarr database will be built.
        """
        self.store_path = store_path
        
        # Open or create the Zarr directory store
        self.root = zarr.open_group(store_path, mode='a')
        
        # Store global metadata attributes at the root of the database
        self.root.attrs['sensor_mapping'] = MAFAULDAMetadata.SENSOR_MAP
        self.root.attrs['dataset_name'] = "MAFAULDA"

        # Use Zarr v3 specific codec (replacing legacy numcodecs)
        # The shuffle parameter is defined strictly as the string 'shuffle' in v3
        self.compressor = BloscCodec(cname='lz4', clevel=5, shuffle='shuffle')

    def write_signal(self, zarr_path: str, data: np.ndarray, attributes: dict):
        """
        Writes a signal matrix in a compressed format and attaches metadata to it.

        Args:
            zarr_path (str): The internal hierarchical path within the Zarr store.
            data (np.ndarray): The numeric tensor data to be serialized.
            attributes (dict): The dictionary containing file-specific metadata to attach.
        """
        
        # Use the new 'compressors' argument and pass the codec within a list
        dataset = self.root.require_array(
            name=zarr_path,
            shape=data.shape,
            chunks=(50000, 8), 
            dtype=data.dtype,
            compressors=[self.compressor] 
        )
        
        # Write the NumPy matrix data into the Zarr array
        dataset[:] = data
        
        # Attach file-specific metadata directly to the array attributes
        dataset.attrs.update(attributes)

# ==========================================
# 4. Ingestion Orchestrator
# ==========================================

class MAFAULDAIngestor:
    """
    Responsibility: Orchestrating all components and executing parallel ingestion across CPU cores.
    """
    
    def __init__(self, raw_data_dir: str, zarr_store_path: str, max_workers: int = 8, dtype: np.dtype = np.float16):
        """
        Configures the master ingestor engine.

        Args:
            raw_data_dir (str): The directory containing the extracted CSV dataset.
            zarr_store_path (str): The target destination for the output Zarr database.
            max_workers (int, optional): Thread pool worker limit. Defaults to 4.
            dtype (np.dtype, optional): Float precision for the data. Defaults to np.float16.
        """
        self.raw_data_dir = raw_data_dir
        self.db = ZarrDatabase(zarr_store_path)
        self.max_workers = max_workers
        self.dtype = dtype

    def _process_single_file(self, file_path: str) -> bool:
        """
        Ingestion pipeline for a single file (designed to run inside an isolated Worker process).

        Args:
            file_path (str): Path to the individual CSV file to process.

        Returns:
            bool: True indicating successful processing.
        """
        # 1. Extract structural and domain metadata
        meta = MAFAULDAMetadata.extract_from_path(file_path, self.raw_data_dir)
        
        # 2. Read the raw CSV matrix data
        data_matrix = CSVReader.read(file_path, dtype=self.dtype)
        
        # 3. Store the array and metadata into the binary Zarr database
        self.db.write_signal(
            zarr_path=meta['zarr_path'],
            data=data_matrix,
            attributes=meta['attributes']
        )
        return True

    def ingest_all(self):
        """
        Scans the entire directory tree and processes all CSV files in parallel.
        """
        # Recursively gather all CSV file paths from the root directory
        file_paths = [str(p) for p in Path(self.raw_data_dir).rglob("*.csv")]
        
        if not file_paths:
            raise RuntimeError(f"No CSV files found in {self.raw_data_dir}")

        print(f"🚀 Found {len(file_paths)} CSV files. Starting parallel ingestion to Zarr...")

        # Utilize ProcessPoolExecutor to max out all available CPU cores without GIL bottlenecks
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_single_file, path): path for path in file_paths}
            
            with tqdm(total=len(file_paths), desc="Ingesting Data") as bar:
                for future in as_completed(futures):
                    try:
                        # Await result to catch and propagate potential inner thread exceptions
                        future.result() 
                    except Exception as e:
                        file_path = futures[future]
                        print(f"❌ Error processing {file_path}: {e}")
                    finally:
                        bar.update(1)
        
        print(f"✅ Ingestion complete. Zarr database saved at: {self.db.store_path}")

# ==========================================
# Performance Optimization Note
# ==========================================
# If you run into thread locking problems or slow parallel execution due to NumPy/Pandas,
# you can enforce single-threaded lower-level libraries by uncommenting these:
#
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"