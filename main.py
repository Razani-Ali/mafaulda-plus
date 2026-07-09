"""
MAFAULDA-PLUS End-to-End Pipeline Demonstration
==============================================
This script orchestrates the complete machine learning data pipeline:
1. Secure multi-threaded downloading.
2. Concurrent CSV parsing & conversion to an optimized Zarr database.
3. Parallel execution of causal IIR filters & signal downsampling.
4. RAM-efficient data loading and generation of native PyTorch DataLoaders.
"""

import os
import numpy as np
import mafaulda

def main():
    # =====================================================================
    # 0. Path Definitions and Initial Configurations
    # =====================================================================
    RAW_ZIP_PATH = "data/MAFAULDA.zip"
    RAW_CSV_DIR = "data/MAFAULDA"          # Extraction directory
    SOURCE_ZARR_PATH = "data/MAFAULDA.zarr"
    FILTERED_ZARR_PATH = "data/MAFAULDA_Filtered.zarr"
    
    # Define target classes and numeric mapping for ML model decoupling
    class_mapping = {
        0: 'normal',
        1: 'imbalance',
        2: 'horizontal-misalignment'
    }
    # Create reverse map for PyTorch DataLoader (string to integer)
    class_to_idx = {v: k for k, v in class_mapping.items()}
    
    print("🏁 Starting MAFAULDA-PLUS Production Pipeline...")

    # =====================================================================
    # 1. Secure Multi-threaded Downloading Phase
    # =====================================================================
    print("\n🌐 [STEP 1] Downloading Dataset...")
    # The download method automatically handles Range Requests and Resume features
    download_success = mafaulda.download(
        target_path=RAW_ZIP_PATH,
        num_connections=8,
        replace=False
    )
    if not download_success:
        print("💀 Pipeline aborted due to download failure.")
        return

    # =====================================================================
    # 2. Parallel and Atomic Archive Extraction Phase
    # =====================================================================
    print("\n📦 [STEP 2] Extracting ZIP Archive Parallelly...")
    from mafaulda.utilities import extract_zip
    extraction_success = extract_zip(
        zip_path=RAW_ZIP_PATH,
        extract_to=RAW_CSV_DIR,
        replace=False,
        max_workers=4
    )
    if not extraction_success:
        print("💀 Pipeline aborted due to extraction failure.")
        return

    # =====================================================================
    # 3. Ingestion & Binary Compression to Zarr v3 Structure
    # =====================================================================
    print("\n⚡ [STEP 3] Ingesting CSV Files into Compressed Zarr Store...")
    # Rapid parsing using C-engine and saving in fp16 precision
    mafaulda.ingest(
        raw_data_dir=RAW_CSV_DIR,
        zarr_store_path=SOURCE_ZARR_PATH,
        max_workers=4,
        dtype=np.float16
    )

    # =====================================================================
    # 4. Causal Signal Filtering & Downsampling Phase (50kHz -> 10kHz)
    # =====================================================================
    print("\n🔮 [STEP 4] Applying Digital Filters & Downsampling...")
    
    # Initialize causal IIR filter for accelerometer signals
    accel_filter = mafaulda.AccelerometerFilter(fs=50000.0, cutoff=6220.0, iir_kind="ellip")
    # Initialize instantaneous RPM computer for tachometer signals
    tacho_processor = mafaulda.TachometerProcessor(Fs=50000, filter_cutoff=400.0)
    
    # Construct mapping for multiprocessing filter engine injection
    filter_config = {
        'Tachometer': tacho_processor,
        'UH Axial Acc': accel_filter,
        'UH Radial Acc': accel_filter,
        'UH Tangential Acc': accel_filter,
        'OH Axial Acc': accel_filter,
        'OH Radial Acc': accel_filter,
        'OH Tangential Acc': accel_filter,
        'Microphone': None  # Microphone channel passes through raw without filtering
    }
    
    # Execute transformations and save into the final processed database
    mafaulda.filter(
        src_store_path=SOURCE_ZARR_PATH,
        dst_store_path=FILTERED_ZARR_PATH,
        filter_map=filter_config,
        downsample_factor=5,  # Reduces sampling frequency by factor of 5 to shorten sequence length
        max_workers=4
    )

    # =====================================================================
    # 5. Base Tensor Loading & Class Filtering for Deep Learning
    # =====================================================================
    print("\n🧠 [STEP 5] Loading ML-Ready Continuous Base Tensors...")
    # Extract data into a balanced cross-validation structure
    X_base, Y_base, meta_base = mafaulda.load(
        zarr_path=FILTERED_ZARR_PATH,
        folds=4,                           # Stratification folds for cross-validation
        labeling_strategy='only types',     # Classification method
        use_memmap=True,                   # Enables disk paging for absolute zero-RAM footprint
        target_classes=list(class_to_idx.keys())  # Load specified target classes only
    )

    # =====================================================================
    # 6. PyTorch Integration & Zero-Copy DataLoader Deployment
    # =====================================================================
    print("\n🔥 [STEP 6] Spawning High-Performance Native PyTorch DataLoader...")
    
    # Build native DataLoader using the zero-copy VirtualWindowing backend
    train_loader = mafaulda.get_pytorch_dataloader(
        X_base=X_base,
        Y_base=Y_base,
        meta_base=meta_base,
        window_size=1024,       # Input window sequence length for neural networks (CNN/RNN)
        step_size=512,          # Overlap increment for the sliding window
        class_to_idx=class_to_idx,
        batch_size=32,
        shuffle=True,           # Shuffles stride views dynamically
        num_workers=2           # Worker threads for PyTorch parallel data loading
    )
    
    # =====================================================================
    # 7. Pipeline Sanity Check Iteration
    # =====================================================================
    print("\n🧪 Running Pipeline Sanity Check Iteration...")
    for batch_idx, (batch_x, batch_y, (batch_sev, batch_rpm)) in enumerate(train_loader):
        print("💥 PyTorch Mini-Batch Successfully Extracted!")
        print(f"   -> Inputs (X) Shape      : {batch_x.shape} (Format: [Batch, Channels, WindowSize])")
        print(f"   -> Targets (Y) Tensor    : {batch_y}")
        print(f"   -> Engine RPM Sample     : {batch_rpm[0].item():.1f} RPM")
        print(f"   -> Severity Metadata     : {batch_sev[0]}")
        break  # Evaluate first step only to verify successful execution
        
    print("\n🎉 All Steps Executed Successfully! Your Enterprise-Grade Dataset Engine is Ready.")

if __name__ == "__main__":
    main()