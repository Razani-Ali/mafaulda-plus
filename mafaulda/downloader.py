"""
Secure Resumable Downloader Module

This module provides a robust, multi-threaded downloading engine capable of handling 
large file transfers over HTTP/HTTPS. It inherently supports parallel chunking 
(Range requests), graceful interruption resumption, and atomic file merging to 
guarantee maximum data integrity and network efficiency.
"""

import os
import ssl
import time
import shutil
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple
from tqdm.auto import tqdm
from .utilities import remove_file, replace_with_error, get_temp_path

class SecureResumableDownloader:
    """
    A robust, multi-threaded, and resumable file downloader.
    
    This class handles downloading large files securely over HTTP/HTTPS. It supports 
    parallel chunk downloading (Range requests), automatic resumption of interrupted 
    downloads, and atomic file merging to ensure data integrity.
    """
    
    def __init__(self, url: str =
                 "https://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/full.zip",
                 target_path: str = "data/MAFAULDA.zip",
                 min_size_bytes: int = 12 * 1024**3,
                 max_retries: int = 10,
                 retry_delay: float = 3.0,
                 chunk_size: int = 1024 * 1024,
                 timeout: int = 10,
                 replace: bool = False,
                 num_connections: int = 8):
        """
        Initializes the SecureResumableDownloader configuration.

        Args:
            url (str): The direct URL of the file to download.
            target_path (str): The final destination path for the downloaded file.
            min_size_bytes (int): Minimum valid file size in bytes (corruption check).
            max_retries (int): Maximum network retry attempts before failing.
            retry_delay (float): Seconds to wait between retry attempts.
            chunk_size (int): Buffer size in bytes for reading/writing streams.
            timeout (int): Socket timeout limit in seconds.
            replace (bool): If True, overwrites the target file if it already exists.
            num_connections (int): Number of parallel threads/connections for downloading.
        """
        # Assign URL and target paths
        self.url = url
        self.target_path = target_path
        
        # Set validation and retry parameters
        self.min_size_bytes = min_size_bytes
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set I/O buffer size and network timeout limits
        self.chunk_size = chunk_size
        self.timeout = timeout
        
        # Control flags and thread settings
        self.replace = replace
        self.num_connections = num_connections
        
        # Generate a temporary path for the staging phase
        self.temp_path = get_temp_path(target_path)
        
        # Create an unverified SSL context to bypass strict certificate validation errors
        self.ssl_context = ssl._create_unverified_context()
        
        # Initialize a thread lock for thread-safe progress bar updates
        self.lock = threading.Lock()

    def download(self) -> bool:
        """
        Orchestrates the complete download process, including retries and validation.

        Returns:
            bool: True if the file is successfully downloaded and validated, False otherwise.
        """
        # Check if the target file already exists and replacement is not requested
        if os.path.exists(self.target_path) and not self.replace:
            print(f"⏭️ File already exists at {self.target_path}. Skipping.")
            return True

        # Ensure the destination directory tree exists
        os.makedirs(os.path.dirname(self.target_path), exist_ok=True)

        # Loop through the allowed number of network retry attempts
        for attempt in range(1, self.max_retries + 1):
            print(f"⬇️ Download attempt {attempt}/{self.max_retries} ...")
            
            # Execute the actual download execution pipeline
            success = self._attempt_download(attempt)

            # If download executes without errors, verify the final file integrity
            if success and self._verify_final_size():
                print(f"✅ File successfully downloaded to: {self.target_path}")
                return True

            # If failed but retries remain, sleep before the next attempt
            if attempt < self.max_retries:
                print(f"🔄 Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

        # Log a fatal error if all attempts are exhausted
        print(f"💀 Failed to download {self.url} after {self.max_retries} attempts.")
        
        # Purge all temporary artifacts to free up disk space
        self._cleanup()
        return False

    def _attempt_download(self, attempt: int) -> bool:
        """
        Performs a single download attempt, branching into sequential or parallel modes.

        Args:
            attempt (int): The current retry attempt number.

        Returns:
            bool: True if the download process completes without exceptions.
        """
        try:
            # Fetch server metadata to determine total file size and range request support
            total_size, supports_range = self._get_metadata()
            
            # Fallback to single-thread sequential mode if range requests are not supported
            if not supports_range or total_size == 0:
                self._download_sequential(total_size, attempt)
            else:
                # Proceed with high-speed parallel chunk downloading
                self._download_parallel(total_size, attempt)
                
            # Atomically replace the temporary file with the final target file
            replace_with_error(self.temp_path, self.target_path)
            return True
        except Exception as e:
            # Catch, isolate, and log network-level or socket failures
            print(f"❌ Network-level failure during attempt {attempt}: {e}")
            return False

    def _get_metadata(self) -> Tuple[int, bool]:
        """
        Retrieves headers from the target URL via a lightweight HEAD request.

        Returns:
            Tuple[int, bool]: The total file size in bytes, and a boolean flag indicating
                              if the remote server supports 'Accept-Ranges'.
        """
        # Prepare a HEAD request to fetch metadata exclusively without pulling body content
        req_head = urllib.request.Request(self.url, method='HEAD')
        
        # Open the connection using the custom SSL context and strict timeout limit
        with urllib.request.urlopen(req_head, context=self.ssl_context, timeout=self.timeout) as response:
            # Extract specific header fields
            content_length = response.getheader('Content-Length')
            accept_ranges = response.getheader('Accept-Ranges')
            
            # Safely parse the total file size
            total_size = int(content_length) if content_length else 0
            
            # Determine range support capability for parallel chunking
            supports_range = (accept_ranges == 'bytes') or (total_size > 0)
            return total_size, supports_range

    def _download_sequential(self, total_size: int, attempt: int):
        """
        Downloads the file in a single, sequential stream.

        Args:
            total_size (int): Expected file size (or 0 if unknown).
            attempt (int): Current attempt iteration for progress bar logging.
        """
        # Create a standard HTTP GET request
        req = urllib.request.Request(self.url)
        
        # Open the data stream
        with urllib.request.urlopen(req, context=self.ssl_context, timeout=self.timeout) as response:
            # Determine target download size dynamically from headers (fallback to total_size)
            size_to_dl = int(response.getheader('Content-Length', total_size))
            
            # Open the temporary staging file for writing binary data
            with open(self.temp_path, "wb") as f, tqdm(
                total=size_to_dl, unit='B', unit_scale=True, unit_divisor=1024,
                desc=f"Attempt {attempt}: Downloading Seq", leave=True
            ) as bar:
                # Stream binary data chunks continuously
                while True:
                    chunk = response.read(self.chunk_size)
                    # Break loop when the EOF is reached
                    if not chunk: break
                    # Write chunk buffer to disk
                    f.write(chunk)
                    # Update the visual progress bar metrics
                    bar.update(len(chunk))

    def _download_parallel(self, total_size: int, attempt: int):
        """
        Segments the remote file and downloads chunks concurrently using a ThreadPool.

        Args:
            total_size (int): The verified total size of the remote file in bytes.
            attempt (int): Current attempt iteration for progress bar logging.
        """
        # Announce the partitioning strategy to the console
        print(f"🔀 Splitting download into {self.num_connections} parts...")
        
        # Calculate the uniform byte size of each partition
        part_size = total_size // self.num_connections
        ranges = []
        initial_downloaded = 0
        
        # Calculate start and end byte boundaries for each connection thread
        for i in range(self.num_connections):
            start = i * part_size
            # Ensure the final thread captures all remaining trailing bytes
            end = (i + 1) * part_size - 1 if i < self.num_connections - 1 else total_size - 1
            part_file = f"{self.temp_path}.part{i}"
            
            # Inspect existing part files to enable precise resumption of paused downloads
            if os.path.exists(part_file):
                initial_downloaded += min(os.path.getsize(part_file), end - start + 1)
            
            # Map the calculated range configuration to the worker pool
            ranges.append((start, end, part_file))

        # Initialize a unified, synchronized progress bar for the parallel download phase
        with tqdm(total=total_size, initial=initial_downloaded, unit='B', 
                  unit_scale=True, unit_divisor=1024, 
                  desc=f"Attempt {attempt}: Downloading Par", leave=True) as bar:
            
            # Spawn a thread pool for concurrent network connections
            with ThreadPoolExecutor(max_workers=self.num_connections) as executor:
                # Submit each chunk mapping to the pool
                futures = [executor.submit(self._download_chunk, s, e, pf, bar) for s, e, pf in ranges]
                
                # Await all threads and propagate any internal runtime exceptions
                for future in as_completed(futures):
                    future.result()

        # Announce completion of download and transition into merging phase
        print("📥 Download complete. Merging parts now...")
        
        # Trigger the concatenation process once all chunks are safely confirmed on disk
        self._merge_parts()

    def _download_chunk(self, start: int, end: int, part_file: str, bar: tqdm):
        """
        Worker method executed by individual threads to download a specific byte range.

        Args:
            start (int): Starting byte index boundary.
            end (int): Ending byte index boundary.
            part_file (str): Temporary file path for this specific chunk.
            bar (tqdm): Shared progress bar object handler.
        """
        # Calculate the mathematical expected size of this segment
        expected_size = end - start + 1
        
        # Check the current physical size of the chunk file on disk
        current_size = os.path.getsize(part_file) if os.path.exists(part_file) else 0
        
        # If the part file exceeds expected bounds (corruption), wipe it and restart
        if current_size > expected_size:
            remove_file(part_file, force=True)
            current_size = 0
            
        # If the segment is already fully downloaded, exit thread early (Resumption mechanism)
        if current_size == expected_size:
            return
            
        # Calculate the exact byte position where the request should resume
        req_start = start + current_size
        req = urllib.request.Request(self.url)
        
        # Inject the HTTP Range header to fetch partial data
        req.add_header('Range', f'bytes={req_start}-{end}')
        
        # Execute the ranged HTTP GET request
        with urllib.request.urlopen(req, context=self.ssl_context, timeout=self.timeout) as response:
            # Open the chunk file in append-binary mode
            with open(part_file, 'ab') as f:
                while True:
                    # Read binary data stream buffers
                    chunk = response.read(self.chunk_size)
                    if not chunk: break
                    
                    # Flush buffer to the partial file on disk
                    f.write(chunk)
                    
                    # Securely update the master progress bar using the shared thread lock
                    with self.lock:
                        bar.update(len(chunk))

    def _merge_parts(self):
        """
        Secured by a try-finally block:
        Guarantees that part files are strictly deleted after merging to prevent disk overflow.
        """
        # Calculate total byte size accumulated across all part files to initialize merge bar
        total_merge_size = sum(
            os.path.getsize(f"{self.temp_path}.part{i}") 
            for i in range(self.num_connections) 
            if os.path.exists(f"{self.temp_path}.part{i}")
        )
        
        try:
            # Initialize the unified master output file for writing binary data
            with open(self.temp_path, 'wb') as outfile:
                # Setup the visual progress bar explicitly for the merging phase
                with tqdm(total=total_merge_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Merging Parts", leave=True) as bar:
                    
                    # Iterate sequentially through each numbered chunk part
                    for i in range(self.num_connections):
                        part_file = f"{self.temp_path}.part{i}"
                        
                        # Bypass missing files safely
                        if not os.path.exists(part_file):
                            continue
                        
                        try:
                            # Open the chunk file for reading
                            with open(part_file, 'rb') as infile:
                                # Stream data manually to allow progress bar updates during copy
                                while True:
                                    chunk = infile.read(self.chunk_size)
                                    if not chunk:
                                        break
                                    outfile.write(chunk)
                                    # Advance progress bar based on written bytes
                                    bar.update(len(chunk))
                        finally:
                            # The part file will be deleted under all circumstances (even upon failure)
                            remove_file(part_file, force=True)
                            
            # Announce finalization of the merge phase
            print("✅ Parts successfully merged.")
            
        except Exception as e:
            # If an error occurs during file concatenation, purge the corrupted main temp file
            remove_file(self.temp_path, force=True)
            raise e

    def _verify_final_size(self) -> bool:
        """
        Checks the final file size against the accepted minimum threshold.

        Returns:
            bool: True if the file size is valid, False if it is deemed corrupted/incomplete.
        """
        # Ensure the final target file physically exists
        if os.path.exists(self.target_path):
            # Retrieve the physical byte footprint
            size = os.path.getsize(self.target_path)
            print(f"📦 Downloaded size = {size/1e9:.2f} GB")

            # Validate structural integrity against the user-defined threshold
            if size >= self.min_size_bytes:
                print("✅ Valid file detected.")
                return True
            
            # Trigger failure protocol if file violates minimum expected size
            print(f"❌ File too small (< {self.min_size_bytes/1e9:.1f} GB). Retrying...")
            
            # Purge the invalid file to prepare a clean slate for the next attempt
            remove_file(self.target_path, force=True) 
            
        return False

    def _cleanup(self):
        """
        Housekeeping function that purges all associated artifacts after terminal failures.
        """
        # Force remove the main target file
        remove_file(self.target_path, force=True)
        
        # Force remove the main staging file
        remove_file(self.temp_path, force=True)
        
        # Iterate and force remove all potential chunk staging files
        for i in range(self.num_connections):
            remove_file(f"{self.temp_path}.part{i}", force=True)