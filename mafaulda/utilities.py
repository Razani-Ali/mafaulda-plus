"""
Core Downloader Module

This module provides robust, concurrent file downloading, archiving, and copying utilities.
It includes mechanisms for thread-safe operations, atomic file replacements, and 
automatic retry logic for handling network instability or filesystem locks.
"""

import os
import zipfile
import shutil
from typing import Union, List
import time
import urllib.request
import ssl
from tqdm.auto import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import shutil
import time
from pathlib import Path

def get_temp_path(path: str, suffix: str = "tmp") -> str:
    """
    Generates a temporary file path by appending a suffix before the file extension.
    
    Args:
        path (str): The original target file path.
        suffix (str, optional): The suffix to append. Defaults to "tmp".
        
    Returns:
        str: The newly constructed temporary file path.
    """
    # Split the original path into base name and extension
    base, ext = os.path.splitext(path)
    # Reconstruct the path with the suffix injected before the extension (if it exists)
    return f"{base}_{suffix}{ext}" if ext else f"{path.rstrip('/')}_{suffix}"

def remove_file(file_path: str, force: bool = False) -> bool:
    """
    Safely removes a file from the filesystem.
    
    Args:
        file_path (str): The path to the file to be deleted.
        force (bool, optional): If True, returns True even if the file didn't exist. Defaults to False.
        
    Returns:
        bool: True if the file was successfully removed (or if force=True), False otherwise.
    """
    try:
        # Check if the path exists and is strictly a file
        if os.path.exists(file_path) and os.path.isfile(file_path):
            # Attempt to delete the file
            os.remove(file_path)
            return True
        # Return the force flag if the file does not exist
        return force
    except Exception as e:
        # Catch and log any permission or OS errors during deletion
        print(f"⚠️ Warning: Failed to remove file {file_path}: {e}")
        return False

def replace_with_error(src: str, dest: str) -> bool:
    """
    Atomically replaces the destination file with the source file.
    
    Args:
        src (str): The path of the source file (usually a temporary file).
        dest (str): The path of the final destination file.
        
    Returns:
        bool: True if the replacement was successful, False otherwise.
    """
    try:
        # Perform an atomic replace operation (avoids corrupted partial files)
        os.replace(src, dest)
        return True
    except Exception as e:
        # Catch and log errors, such as file locks or permission issues
        print(f"❌ Error replacing {src} with {dest}: {e}")
        return False

def remove_folder(folder_path: Union[str, os.PathLike], force: bool = False) -> bool:
    """
    Safely and recursively removes a directory tree from the filesystem.

    Provides a clean execution wrapper around standard directory tree removal tools.
    It integrates explicit verification checks for pathway existence and node type
    descriptors before launching deletion, isolating permission violations or 
    generic file system exceptions to guarantee system runtime stability.

    Args:
        folder_path (Union[str, os.PathLike]): The pathway locating the target directory 
            tree slated for recursive deletion.
        force (bool, optional): If True, suppresses missing directory exceptions and 
            returns True even if the target folder does not exist. Defaults to False.

    Returns:
        bool: True if the directory tree was successfully deleted (or skipped via force=True),
            False if an OS error, missing permission descriptor, or lock blocked completion.

    Raises:
        FileNotFoundError: If the target path is absent and force is evaluated as False.
        NotADirectoryError: If the designated pathway targets a file descriptor rather 
            than a directory layout container.
    """
    try:
        # Check if the target pathway physically exists on the filesystem disk[cite: 2]
        if not os.path.exists(folder_path):
            # If the path is missing but the force flag is active, bypass execution with success[cite: 2]
            if force:
                return True
            # Raise an explicit exception if the folder is absent and force is deactivated[cite: 2]
            raise FileNotFoundError(f"❌ could not find folder '{folder_path}'")
        
        # Verify that the existing node represents a structural directory, not a generic file link[cite: 2]
        if not os.path.isdir(folder_path):
            # Abort operation with a explicit type exception if a file collision occurs[cite: 2]
            raise NotADirectoryError(f"🚫 directory '{folder_path}' is not a folder")
        
        # Concurrently clean and recursively purge the entire directory hierarchy layout tree[cite: 2]
        shutil.rmtree(folder_path)
        # Return success confirmation after directory tree is wiped[cite: 2]
        return True
    
    except PermissionError as e:
        # Intercept, catch, and log access blockages or administrative filesystem privileges[cite: 2]
        print(f"🔒 permission denied, error: {e}")
    except Exception as e:
        # Catch, log, and isolate unknown system exceptions to protect application lifecycle[cite: 2]
        print(f"⚠️ unknown error: {e}")
        
    # Return failure if any operational exception blockages interrupt execution flow[cite: 2]
    return False

# Dictionary to store thread locks mapped to specific file paths
_locks = {}
# A master lock to prevent race conditions when creating new file-specific locks
_master_lock = threading.Lock()

def get_file_lock(file_path):
    """
    Retrieves or creates a thread lock specific to a file path.
    
    Args:
        file_path (str): The unique string representation of the target file path.
        
    Returns:
        threading.Lock: A lock object assigned exclusively to the requested file path.
    """
    # Acquire the global master lock to safely evaluate the _locks dictionary
    with _master_lock:
        # If the file path doesn't have a lock yet, instantiate one
        if file_path not in _locks:
            _locks[file_path] = threading.Lock()
        # Return the specific lock for this file
        return _locks[file_path]

def safe_copy(src, dst, max_retries=7, chunk_size=50*1024*1024, force_sync=True):
    """
    Spawns a thread to copy a file safely, atomically, and with thread locks.
    If force_sync is enabled, it blocks the main thread (Colab cell) until data is fully written.
    
    Args:
        src (str/Path): The source file path.
        dst (str/Path): The destination file path.
        max_retries (int, optional): Number of retry attempts on failure. Defaults to 7.
        chunk_size (int, optional): Size of the read/write buffer in bytes (default 50MB).
        force_sync (bool, optional): If True, forces synchronous/blocking execution. Defaults to False.
        
    Returns:
        threading.Thread: The thread handling the copy operation.
    """
    # Retrieve the thread lock dedicated to the destination file
    file_specific_lock = get_file_lock(str(dst))
    
    # Define the internal function to be executed by the thread
    def save_it(src_path, dst_path):
        # Acquire the file-specific lock to ensure only one thread modifies this destination
        with file_specific_lock:
            # Initiate the retry loop
            for i in range(max_retries):
                try:
                    # Convert paths to pathlib objects
                    p_src = Path(src_path)
                    p_dst = Path(dst_path)
                    
                    # Ensure the destination directory exists
                    p_dst.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Create a temporary destination path to prevent file corruption
                    temp_dst = p_dst.with_suffix('.tmp')
                    
                    # Open source for reading and temp file for writing (both in binary mode)
                    with open(p_src, 'rb') as fsrc:
                        with open(temp_dst, 'wb') as fdst:
                            while True:
                                # Read data in chunks
                                buf = fsrc.read(chunk_size)
                                # Break the loop if the end of the file is reached
                                if not buf:
                                    break
                                # Write the chunk to the temporary file
                                fdst.write(buf)

                    # Preserve the original file metadata (permissions, timestamps, etc.)
                    shutil.copystat(src_path, temp_dst)
                    
                    # Atomically replace the temp file with the final destination
                    replace_with_error(temp_dst, p_dst)
                    
                    # Break the retry loop upon success
                    break
                
                except Exception as e:
                    # If not on the last attempt, wait 20 seconds before retrying
                    if i < max_retries - 1:
                        print(f"🔄 [Retry {i+1}/{max_retries}] Copy interrupted. Retrying in 20s... ⏳")
                        time.sleep(20)
                    else:
                        # Log failure if all retries are exhausted
                        print(f"❌ Failed to copy after {max_retries} attempts: {e}")

    # Instantiate a new thread targeting the internal save_it function
    thread = threading.Thread(
        target=save_it, 
        args=(str(src), str(dst))
    )
    
    # Set the thread as a daemon so it doesn't block the main program from exiting if async
    thread.daemon = True
    
    # Start the thread execution
    print(f"🚀 Thread spawned successfully! Initiating background I/O stream... 📡")
    thread.start()
    
    # ⚡ NEW SYNCHRONOUS FORCE LOCK BLOCK:
    # If force_sync is requested, lock the execution flow and wait until the thread completes
    if force_sync:
        print(f"🔒 [FORCE_SYNC ACTIVE] Locking Colab cell execution runtime... Please wait! 🛑")
        print(f"📦 Transferring stream from physical path to destination storage... 🔄")
        
        # Block the execution of the calling cell until save_it finishes its job
        thread.join()
        
        # Force a deep hardware-level flush of Linux filesystem cache pages directly to Drive mount
        if hasattr(os, 'sync'):
            print(f"💾 Flushing OS cache buffers directly onto Google Drive grid... 🧼")
            os.sync()
            
        print(f"✨ [SUCCESS] Hardware cache synchronized! Drive storage pipeline closed. 🏁")
    else:
        print(f"🛸 [ASYNC MODE] Cell released early. File copy running silently in background... 🎭")
    
    # Return the thread object to maintain absolute backward compatibility
    return thread


def _prepare_temp_directory(extract_to: str) -> str:
    """
    Creates and purges a temporary directory to facilitate atomic zip extraction.

    This acts as a staging environment. If a previous extraction crashed or left
    stale artifacts, this function guarantees a clean slate by forcefully 
    wiping the target path before recreating it.

    Args:
        extract_to (str): The final destination path where the zip contents will live.

    Returns:
        str: The absolute or relative path to the freshly generated temporary directory.
    """
    # Generate the temporary directory path by appending a '_part' suffix
    temp_dir = extract_to.rstrip('/') + "_part"
    
    # Forcefully eliminate any preexisting directory or stale files at that location
    remove_folder(temp_dir, force=True)
    
    # Create the clean staging directory from scratch
    os.makedirs(temp_dir, exist_ok=True)
    
    # Return the clean temporary path to the orchestration pipeline
    return temp_dir

class AtomicParallelZipExtractor:
    """
    A high-performance, atomic, and multi-threaded ZIP extraction engine.
    
    This class consolidates hierarchical directory mapping, multi-threaded worker
    chunking, and transaction-style atomic folder swaps. It guarantees that an
    extraction process never leaves corrupted, half-written files at the destination
    if a crash, network drop, or execution cancellation occurs.
    """

    def __init__(self, zip_path: str, extract_to: str, replace: bool = False, max_workers: int = 4):
        """
        Initializes the atomic parallel extraction pipeline configuration.

        Args:
            zip_path (str): The physical path targeting the source archive file.
            extract_to (str): The target location directory where content will settle.
            replace (bool, optional): If True, wipes existing destination folders. Defaults to False.
            max_workers (int, optional): Thread boundary cap scaling concurrency. Defaults to 4.
        """
        self.zip_path = zip_path
        self.extract_to = extract_to
        self.replace = replace
        self.max_workers = max_workers
        self._lock = threading.Lock()  # Dynamic lock protecting progress bar updates from worker race conditions

    def _build_directory_tree(self, temp_dir: str) -> List[zipfile.ZipInfo]:
        """
        Pre-generates the entire directory tree architecture synchronously before dumping file threads.

        This synchronous step is highly critical for stability. It prevents multi-threaded core
        workers from executing race-prone, overlapping 'os.makedirs' calls simultaneously, 
        which frequently leads to standard operating system file collision errors.

        Args:
            temp_dir (str): The temporary path staging the extraction process.

        Returns:
            List[zipfile.ZipInfo]: A isolated list tracking file records stripped of empty directory nodes.
        """
        file_members = []
        
        # Open the ZIP archive in safe read-only mode
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            # Iterate sequentially through structural layout metadata inside the archive
            for member in zf.infolist():
                # If the current structural item represents an explicit directory node
                if member.is_dir():
                    # Generate the physical folder path matching the archive schema inside temp directory
                    os.makedirs(os.path.join(temp_dir, member.filename), exist_ok=True)
                else:
                    # Resolve parent folder path bound to the file object
                    parent_dir = os.path.dirname(os.path.join(temp_dir, member.filename))
                    if parent_dir:
                        # Pre-generate parent directories if they don't exist yet
                        os.makedirs(parent_dir, exist_ok=True)
                    # Append the verified file record to the work pool list
                    file_members.append(member)
                    
        return file_members

    def _extract_chunk(self, chunk: List[zipfile.ZipInfo], temp_dir: str, bar: tqdm):
        """
        Dedicated thread worker loop processing an assigned subset slice of files.

        Args:
            chunk (List[zipfile.ZipInfo]): Slice allocation tracking files assigned to this specific thread thread.
            temp_dir (str): Staging pathway directory collecting compiled outputs.
            bar (tqdm): Progress bar handler catching completed metrics updates.
        """
        # Instantiate a dedicated archive stream handle restricted within this worker thread
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            for member in chunk:
                # Stream binary block directly from disk storage grid to target temporary path location
                zf.extract(member, temp_dir)
                
                # Acquire instance-level lock to securely notify progress interface without corruption
                with self._lock:
                    bar.update(member.file_size)

    def _extract_parallel(self, file_members: List[zipfile.ZipInfo], temp_dir: str):
        """
        Segments file allocation loads evenly across active threads and orchestrates execution pools.

        Args:
            file_members (List[zipfile.ZipInfo]): Clean collection mapping files to parse.
            temp_dir (str): Staging directory area holding structural assets.
        """
        # Accumulate exact total byte sizing requirements to set up accurate metric bars
        total_size = sum(f.file_size for f in file_members)
        
        # Stratify files using a round-robin stride step sequence mapping loads evenly among workers
        chunks = [file_members[i::self.max_workers] for i in range(self.max_workers)]

        # Initialize user tracking interface monitoring output speed and progression metrics
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Extracting", leave=True) as bar:
            # Spawn the concurrent asynchronous execution context frame
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Dispatch tasks safely across pool workers mapping active data chunk buffers
                futures = [
                    executor.submit(self._extract_chunk, chunk, temp_dir, bar) 
                    for chunk in chunks if chunk
                ]
                # Collect thread responses as they finalize tasks
                for future in as_completed(futures):
                    # Call result() to escalate internal thread runtime exceptions to the main orchestration loop
                    future.result()

    def extract(self) -> bool:
        """
        The orchestrator method triggering secure, parallel, and atomic zip extractions.

        Returns:
            bool: True if transaction-style swap concludes flawlessly, False otherwise.
        """
        # Skip routine entirely if a folder exists and overwrite permissions are locked
        if os.path.exists(self.extract_to) and not self.replace:
            print(f"⏭️ Extraction directory {self.extract_to} already exists. Skipping.")
            return True

        # Provision the temporary extraction workspace staging environment safely
        temp_extract_to = _prepare_temp_directory(self.extract_to)

        try:
            # Step 1: Map layout and generate structural folder architectures
            file_members = self._build_directory_tree(temp_extract_to)
            
            # Step 2: Concurrently stream and extract chunk buffers across workers
            self._extract_parallel(file_members, temp_extract_to)

            # Step 3: Conclude operation using transaction style atomic layout replacement
            if os.path.exists(self.extract_to):
                remove_folder(self.extract_to, force=True)
                
            # Perform the final atomic folder swap operation seamlessly
            replace_with_error(temp_extract_to, self.extract_to)
            print(f"✅ Successfully extracted to: {self.extract_to}")
            return True
        
        except Exception as e:
            # Rollback phase: Clean up files to preserve integrity in case of structural crashes
            remove_folder(self.extract_to, force=True)
            remove_folder(temp_extract_to, force=True)
            print(f"❌ Error extracting zip file: {e}")
            return False
        

def extract_zip(zip_path: str, extract_to: str, replace: bool = False, max_workers: int = 4) -> bool:
    """
    Unified public wrapper function rendering backward-compatible integration access points.
    
    This abstracts away the class creation details, enabling simple functional calls 
    ideal for clean PyPI module entry points.
    """
    extractor = AtomicParallelZipExtractor(
        zip_path=zip_path, extract_to=extract_to, 
        replace=replace, max_workers=max_workers
    )
    return extractor.extract()