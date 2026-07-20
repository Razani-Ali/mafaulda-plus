<div align="center">

# 🚀 MAFAULDA-Plus: The Zero-RAM Machinery Dataset Engine

**An enterprise-grade, concurrent, and zero-copy ingestion, filtering, and windowing pipeline for the massive MAFAULDA machinery fault diagnosis dataset.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_QO6B5rM79knfOL3ghE3jFoH533yIVwb?usp=sharing)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/mafaulda-plus.svg)](https://badge.fury.io/py/mafaulda-plus)

</div>

---

## 🌟 The Big Data Challenge, Conquered
Following the massive success of Bearing Fault Dataset, **CWRU**, we realized that modern Deep Learning researchers needed an even more powerful engine. While the CWRU dataset is manageable in size, the MAFAULDA dataset is an absolute beast—dozens of gigabytes of raw, multi-channel vibration signals. 

Attempting to load this into RAM using traditional methods will instantly crash your low RAM system. **MAFAULDA-Plus** is engineered from the ground up to handle this exact nightmare. By leveraging **Zarr v3 compression**, **disk-backed Memory Mapping (`memmap`)**, and a purely **Virtual Windowing Engine**, we have reduced the RAM consumption of processing the entire dataset to **Absolute Zero**. 

### 🔥 Key Superpowers
*   **🧠 Absolute Zero-RAM Footprint:** Train Deep Learning models on massive arrays without consuming a single byte of physical RAM until the exact moment a mini-batch is dispatched to the GPU.
*   **⚡ C-Engine & Multi-Threading:** We bypass Python's GIL. From parallel downloading to C-backend Pandas CSV reading, your CPU cores will be maxed out for unprecedented speed.
*   **🎯 Advanced Few-Shot Sampler:** A built-in, highly strategic `FewShotSampler` designed specifically for Meta-Learning, allowing you to instantly extract balanced or imbalanced task episodes.
*   **🪄 The 5-Line Miracle:** Go from a web URL to a native PyTorch DataLoader in just 5 lines of code.

---

## 🚀 Quick Start: The "5-Line" Miracle

Forget writing complex I/O loops and chunking logic. We have abstracted the entire data engineering pipeline. Here is how you download, extract, ingest, memory-map, and deploy a virtual sliding window straight into PyTorch:

```python
import mafaulda

# 💡 NOTE: If you already have the dataset downloaded or extracted locally, 
# you can safely skip Step 1 and/or Step 2 to bypass redundant I/O operations.
# 1 & 2. Download and Extract the massive dataset safely via multi-threading
mafaulda.download(target_path="data/MAFAULDA.zip")
mafaulda.utilities.extract_zip(zip_path="data/MAFAULDA.zip", extract_to="data/MAFAULDA", max_workers=8)

# 3. Ingest raw CSVs into a highly compressed Zarr v3 database via multi-processing
mafaulda.ingest(raw_data_dir="data/MAFAULDA", zarr_store_path="data/MAFAULDA.zarr")

# 4. Load a Zero-RAM Memmap tensor (Filtering specific domains instantly!)
X, Y, meta = mafaulda.load(zarr_path="data/MAFAULDA.zarr", target_classes=['normal', 'imbalance'], use_memmap=True)

# 5. Spawn a Native PyTorch DataLoader using our Zero-Copy Virtual Windowing engine
dataloader = mafaulda.get_pytorch_dataloader(X, Y, window_size=2048, step_size=512, class_to_idx={'normal':0, 'imbalance':1})
```

---

## ⚙️ Installation

You can install **MAFAULDA-Plus** directly from PyPI. It automatically manages heavy dependencies like `zarr`, `pandas`, `numpy`, and `scipy`:

```bash
pip install mafaulda-plus
```

---

## 🔬 Advanced: Domain Filtering (A Massive Upgrade)

In our previous library, **CWRU-Plus**, researchers had to load the entire dataset and use NumPy boolean masking (`X[source_mask]`) to isolate specific domains.

**MAFAULDA-Plus completely revolutionizes this.** To maximize memory efficiency, domain filtering now happens *before* the memory map is even allocated! You can strictly target specific classes, RPM ranges, or severities directly inside the `load()` function. Unwanted domains are never processed, saving you massive amounts of disk I/O and time:

```python
import mafaulda

# Load ONLY the specific domain you need for Domain Adaptation research
X, Y, meta = mafaulda.load(
    zarr_path="data/MAFAULDA.zarr",
    target_classes=['misalignment', 'imbalance'],  # Ignore 'normal' or 'overhang'
    target_severities=['20g', '35g'],              # Isolate specific intensities
    rpm_range=(1000.0, 3000.0),                    # Filter by operational speed bounds
    use_memmap=True                                # Keep it on the disk!
)
```

---

## 💾 Optimization: Memory Mapping vs. In-RAM Physical Extraction

By default, **MAFAULDA-Plus** utilizes a zero-copy memory-mapped virtual view (`use_memmap=True`) to handle massive multidimensional signal arrays without blowing up your system's RAM. 

However, if your workstation or server is equipped with a high-capacity RAM pool (e.g., **32 GB or higher**) and you are not concerned about allocating ~20 GB for high-speed computation, you can disable memory mapping. By setting `use_memmap=False`, the entire continuous tensor is pulled into RAM, and you can pass it to `SlidingWindow` to perform instantaneous physical window copies at C-speed. 

**💡 Pro-Tip:** Disabling memory mapping when you have sufficient RAM completely eliminates continuous disk read/write cycles, significantly accelerating your training speed and **extending the lifespan of your SSD/NVMe drive**.

```python
import mafaulda

# 1. Pull the continuous base tensors directly into physical RAM (Requires ~20GB free RAM)
X_base, Y_base, meta_base = mafaulda.load(
    zarr_path="data/MAFAULDA.zarr",
    use_memmap=False  # Disabled to protect SSD lifespan and maximize speed!
)

# 2. Extract explicit, high-speed physical window slices directly in-memory via C-backend
X_phys, Y_phys, meta_phys = mafaulda.SlidingWindow(
    X_base=X_base, 
    Y_base=Y_base, 
    meta_base=meta_base,
    window_size=2048, 
    step_size=512
)
```

---
## 🛡️ Leakage-Free Dataset Splitting & Multi-Framework Deployment
In machinery fault diagnosis (such as CWRU or MAFAULDA benchmarks), performing a random split after applying a sliding window creates catastrophic Data Leakage. Because adjacent windows overlap, the model will essentially memorize the time-series segments during training, leading to artificially inflated test accuracies that fail in real-world validation.

To maintain scientific integrity, MAFAULDA-Plus enforces strict File-Level Stratified Splitting. This ensures that:

*   1. All overlapping windows originating from the same physical file are contained within the exact same partition (Train, Val, or Test).

*  2. The global statistical class distribution is perfectly maintained across all splits without moving or copying a single byte in memory.

🚀 Minimal Production Example
Here is how you can load the massive base tensors, partition the files cleanly, and spin up production pipelines for PyTorch, TensorFlow, and Few-Shot samplers with a near-zero RAM footprint:

```python
import mafaulda

# 1. Memory-Map Database Tensors (Zero-RAM Overhead Loading)
X_base, Y_base, meta_base = mafaulda.load(
    zarr_path="data/MAFAULDA.zarr")

# 2. File-Level Stratified Partitioning (Strict Experiment Isolation)
tr_idx, val_idx, te_idx = mafaulda.stratified_file_split(
    y=Y_base, 
    train_ratio=0.7, 
    val_ratio=0.2, 
    random_seed=42 
)

# 🔥 You can Pass Splitting Indices to Any Function That has 'valid_files' Argument
# e.g. SlidingWindow, sample_few_shot_tasks, ... like 'valid_folds' argument
train_loader_torch = mafaulda.get_pytorch_dataloader(
    X_base=X_base, Y_base=Y_base
    window_size=1024, step_size=512, valid_files=tr_idx)

```

⚠️ **WARNING ON DISTRIBUTIONAL LEAKAGE:**

While restricting fold allocations via the `valid_folds` parameter guarantees zero exact data-point overlap (preventing direct sample leakage), users must exercise extreme caution.
``
Because structural vibration signals exhibit strong quasi-stationarity and temporal correlation over long continuous runs, splitting a single physical signal time-series into sequential folds (e.g., using fold 1 for training and fold 2 for validation) can introduce severe Distributional Leakage (Covariate Shift). The statistical characteristics of contiguous time segments remain highly similar, which can artificially inflate validation metrics and lead to overly optimistic performance estimates.

For statistically rigorous benchmarking, it is strongly recommended to perform cross-validation strictly by shuffling independent physical files via `valid_files` rather than relying solely on sequential temporal sub-segment folds.

---

## 🔬 Advanced: Scientific Digital Signal Processing (DSP)

In real-world condition monitoring, naive data loading introduces catastrophic bottlenecks. **MAFAULDA-Plus** addresses two of the most crucial, yet frequently ignored, signal processing pitfalls in machinery fault diagnosis:

### 1. The Tachometer Overfitting Trap 🎯
Many researchers naively feed raw, noisy tachometer pulse signals directly into deep neural networks. Because high-frequency noise corrupts the pulse edges, deep models easily overfit to transient noise spikes rather than the underlying rotational mechanics. Our **`TachometerProcessor`** provides an enterprise-grade, single-responsibility solution: it applies low-pass filtering, robust percentile scaling, and custom rising-edge separation to dynamically calculate the precise, smooth instantaneous RPM vector.

### 2. Piezoelectric Accelerometer Resonance Interference ($23\text{ kHz}$) 🛡️
The physical accelerometers used to record the MAFAULDA dataset exhibit a notorious high-frequency resonance/interference peak centered around **$23\text{ kHz}$**. Feeding this un-attenuated resonance into a CNN or Vision Transformer forces the model to heavily overfit to the sensor's internal structural artifacts rather than the actual rolling element degradation. Our **`AccelerometerFilter`** uses a highly aggressive, causal Elliptic/Butterworth IIR filter to cut off these parasitic frequencies, enforcing strict generalization across cross-domain environments.

Here is how you inject this scientific pipeline concurrently across all CPU cores:

```python
import mafaulda
# 🚀 Clean and explicit submodule import directly from the unified engine package
from mafaulda.filter_config import TachometerProcessor, AccelerometerFilter

# 🚀 Step 1: Initialize the advanced Tachometer Pulse Edge-Differentiator
tacho_processor = TachometerProcessor(
    Fs=50000, 
    filter_cutoff=400.0,     # Clean high-frequency noise from pulse streams
    pulses_per_rev=1
)

# 🚀 Step 2: Initialize the causal IIR filter to eliminate the 23kHz sensor resonance
accel_filter = AccelerometerFilter(
    fs=50000.0, 
    cutoff=6220.0,           # Safe cut-off well below the 23kHz parasitic resonance peak
    iir_kind="ellip", 
    iir_order=15
)

# Step 3: Map custom callables to channels (Inversion of Control)
filter_pipeline = {
    'Tachometer': tacho_processor,
    'UH Axial Acc': accel_filter,
    'UH Radial Acc': accel_filter,
    'UH Tangential Acc': accel_filter,
    'OH Axial Acc': accel_filter,
    'OH Radial Acc': accel_filter,
    'OH Tangential Acc': accel_filter,
    'Microphone': None        # Keep raw acoustic emissions intact
}

# Step 4: Execute multi-processed filtering and decimation across the Zarr grid
mafaulda.filter(
    src_store_path="data/MAFAULDA.zarr",
    dst_store_path="data/MAFAULDA_Filtered.zarr",
    filter_map=filter_pipeline,
    downsample_factor=1,      # Retain original sampling rate or downsample if needed
    max_workers=8
)
```

---

## 🎯 Advanced: Meta-Learning & Few-Shot Sampling

If you are researching Prototypical Networks, MAML, or Few-Shot Fault Diagnosis, building episodic tasks is usually a nightmare. MAFAULDA-Plus features a dedicated `FewShotSampler` that completely decouples strings from your ML logic.

Just pass your class indices and request your N-way K-shot samples:

```python
import mafaulda

# Map your numeric Deep Learning classes to the database strings
my_class_map = {0: 'normal', 1: 'imbalance', 2: 'misalignment'}

# Request 50 samples for class 0, and only 5 samples for class 2 (Imbalanced Task)
target_ids = (0, 2)
task_counts = (50, 5)

# Initialize the episodic sampler framework by injecting numeric maps and constraints
sampler = mafaulda.sample_few_shot_tasks(
    X_base=X, Y_base=Y, 
    numeric_to_string=my_class_map,
    window_size=1024, step_size=512,
    seed=2026 # 100% Reproducible episodes!
)

# Instantly retrieve your isolated episode containing structured integers for CrossEntropyLoss
X_task, Y_task, _ = sampler.sample(
    target_numeric_classes=target_ids, 
    samples_per_class=task_counts,
)
```

---

# 🛠️ Zero-RAM Streaming Feature Engineering Pipeline

Processing raw, high-frequency machinery diagnostics datasets presents a classic engineering trade-off: **handcrafted statistical libraries (like TSFEL) provide immense diagnostic value but impose devastating computational bottlenecks during training loops.**

`mafaulda-plus` resolves this by introducing a highly optimized **Zero-RAM Feature Extraction Pipeline**. By transforming continuous, multi-fold raw signals directly onto disk-backed grid architectures (`Zarr`), it allows you to run expensive feature extraction **exactly once**, bypassing multiprocessing overheads and unlocking microsecond downstream data recovery for PyTorch and TensorFlow.


### 🚀 Key Advantages

* **Immense Temporal Recovery:** Extracting rich time-series domains (statistical, temporal, and spectral features via TSFEL) takes roughly **0.1 seconds per window**. For large-scale diagnostics arrays, sequential training will quickly choke your CPU. This pipeline executes the stream once, flushes it to disk, and allows subsequent ML iterations to load the complete multi-dimensional feature space instantly.
* **Dynamic Dimension Agnostic Views:** The injected `transform_fn` is completely decoupled from the data pipeline structure. It can return a NumPy array of **any shape or dimensionality** (e.g., standard flat feature vectors, or cross-channel covariance matrices). The framework automatically adapts its disk chunks to preserve your configuration.
* **Leakage-Free Architectural Integrity:** Metadata tracking parameters (`Labels`, `Severity`, `RPM`) are automatically broadcasted to match the exact generated window grids. Data partitioning is enforced strictly at the physical file level, completely preventing cross-validation data leakage or covariate shifts.

### 🛠️ Minimal Production Example

Here is how easily you can stream, archive, and materialize a clean `PyTorch` or `TensorFlow` machine learning dataset with minimal code footprint:

```python
import mafaulda
import tsfel
import numpy as np

# 1. Access the optimal TSFEL extraction configuration
cfg = tsfel.get_features_by_domain("statistical")

# 2. Define your hybrid feature extractor (Accepts shape: [Channels, Length])
def machinery_tsfel_extractor(window_2d: np.ndarray) -> np.ndarray:
    features = []
    # Extract features channel-by-channel
    for ch in range(window_2d.shape[0]):
        df = tsfel.time_series_features_extractor(cfg, window_2d[ch], fs=50000, verbose=0)
        features.extend(df.iloc[0].values)
    
    # Inject cross-channel interaction matrix (e.g., Covariance)
    features.append(np.cov(window_2d[0], window_2d[1])[0, 1])
    return np.array(features)

# 3. Stream & Compile to Zarr on disk with absolute ZERO memory leakage
zarr_root = mafaulda.feature_extraction_pipeline(
    X_raw=X_base, Y_raw=Y_base, meta_raw=meta_base,
    window_size=1024, step_size=512,
    transform_fn=machinery_tsfel_extractor,
    save_zarr_path="data/engineered_features.zarr",
    sensor_names=['UH Axial Acc',
    'the rest of sensors passed to mafaulda.load argument: selected_sensors']
)

# 4. Filter and memory-map the heavy array on disk while lightweight meta hits RAM
X_feat, Y_feat, meta_feat = mafaulda.load_zarr_to_tensor(
    zarr_path="data/engineered_features.zarr",
    as_memmap=True,
    memmap_path="data/final_features.dat",
    selected_sensors='all',
    target_classes=['normal', 'imbalance']
)

# 5. Spawn immediate PyTorch Dataloader for downstream Few-Shot or Deep Learning models
torch_loader = mafaulda.get_pytorch_feature_loader(
    X_base=X_feat, Y_base=Y_feat, meta_base=meta_feat,
    class_to_idx={'normal': 0, 'imbalance': 1},
    batch_size=32, shuffle=True,
    valid_folds=[0, 1, 2], valid_files=[0, 1, 2, 3, 4]
)
# get_tensorflow_featureset & sample_few_shot_features are tailored for this, as well as get_pytorch_feature_loader

```



### ⚠️ Performance Anti-Pattern Alert

> Do **NOT** use this offline workspace to pre-compute and store massive 2D or 3D image-like representations (such as massive Continuous Wavelet Transform (CWT) matrices or high-resolution STFT spectrograms) for deep learning models. Storing millions of redundant floating-point matrix coordinates on disk will drastically inflate file sizes and severely bottle your storage I/O bandwidth during batch ingestion.> **The Right Way:** For heavy spectral image matrices, store the raw compressed time-domain arrays via `mafaulda-plus` and enforce the spatial transformations **on-the-fly** inside the PyTorch/TensorFlow pipeline on a per-batch basis utilizing GPU operations. Keep this offline workspace exclusively reserved for **statistical, handcrafted, or lightweight tabular feature blocks (like TSFEL)**.

---

## 💡 Pro Tip: Multi-Domain Continuum via Disk-Streamed Concatenation

In real-world rotordynamic diagnostics, training robust cross-domain architectures requires merging distinct operational regimes (e.g., separating low-speed and high-speed motor dynamics into a unified training continuum). 

Instead of blowing up your system RAM by loading and stacking numpy arrays in-memory, `mafaulda-plus` allows you to stream out-of-core memory-mapped tensors and physically unify them directly on your hard drive with absolute zero memory overhead.

Here is how you can seamlessly stitch multi-domain RPM datasets under a compact code footprint:

```python
import mafaulda

# 1. Load two heavy, distinct operational domains via zero-RAM memory maps
X_domain1, _, _ = mafaulda.load(
    zarr_path="data/mafaulda_master.zarr",
    rpm_range=(600, 1200),
    target_classes=['normal', 'imbalance'],
    use_memmap=True
)

X_domain2, _, _ = mafaulda.load(
    zarr_path="data/mafaulda_master.zarr",
    rpm_range=(1800, 2400),
    target_classes=['normal', 'imbalance'],
    use_memmap=True
)

print(f"📡 Domain 1 Files: {X_domain1.shape[1]} | Domain 2 Files: {X_domain2.shape[1]}")

# 2. Concurrently stream and forge a unified memory-mapped domain on disk
X_unified_memmap = mafaulda.disk_streamed_concat(
    tensors=(X_domain1, X_domain2),
    axis=1,         # Contiguously stitch along the physical File axis
    output_path="data/multi_domain_unified.dat",
    chunk_size=5    # Isolated transactional chunk caching bounds
)

print(f"📊 Unified Master Shape: {X_unified_memmap.shape}")
# Ready to be wrapped by VirtualFeatureWindow or downstream framework dataloaders!
```

---
## 🚀 Performance Benchmarks & Core Advantages

This library is engineered to modernize data pipelines for massive industrial time-series datasets. It completely eliminates traditional bottlenecks such as Out-Of-Memory (OOM) crashes, cloud storage freezing, and CPU-bound digital signal processing.

### ⏱️ One-Time Setup, Lifetime Speed: Parallel Ingestion & Persistence (Colab Benchmark)
Traditional pipelines parsing thousands of nested CSVs and applying digital filters take hours on every single run. Our architecture enforces a strict **"Ingest Once, Restore Instantly"** philosophy:
* **High-Speed Ingestion:** Safely parsed and compressed 1,951 raw CSV files (31.04 GB) into a highly optimized Zarr binary database (7.27 GB) in just **~11 minutes**.
* **Blazing-Fast Signal Processing:** Executed parallel scientific filtering and signal decimation across all 1,951 multidimensional arrays (8 signals per file) in **only 3 minutes and 13 seconds**.
* **Seamless Pipeline Persistence:** Once this initialization is complete, you never have to run it again. The entire compressed environment is cached globally. Future sessions bypass the raw download, extraction, and filtering stages entirely, restoring the full multi-gigabyte ready-to-train database layout in **under 4.5 minutes**!

### 🧠 Zero-RAM Architecture: Free-Tier Friendly
Handling massive time-series data—especially when applying sliding windows—typically causes physical memory to explode, crashing standard local machines and free-tier cloud notebooks. We solved this preemptively:
* **Lazy Tensor Mapping:** Loaded the complete dataset into the pipeline with a measured physical RAM growth of only **~7.5 MB**. The data remains securely mapped to the disk layout.
* **Virtual Sliding Windows:** Instead of duplicating data into RAM to create sliding windows (which rapidly inflates memory usage to tens of gigabytes), our engine creates **Zero-Copy Virtual Windows**. The 10+ GB virtual memory footprint is handled entirely downstream!
* **Instant Native DataLoaders:** Deep learning data pipelines yield their first mini-batch in **~20 ms (PyTorch)** and **~62 ms (TensorFlow)**, keeping your RAM completely free for model weights and GPU tensors.

---

## ☁️ Google Colab Integration & Real-World Benchmarks

Training on the cloud? MAFAULDA-Plus is fully optimized for Google Colab environments and local workstations alike. 

While perfectly tailored to max out multi-core CPU threads on local machines, it features specialized environment-agnostic defenses to conquer the notorious instability of cloud notebook environments:
* **FUSE-Stabilized Cloud Sync:** Bypasses Google Drive's FUSE file-creation limits and freezing issues by utilizing an on-the-fly parallel archiving pipeline. It safely packs and syncs the entire 7.27 GB database to your cloud drive in **~3 minutes**, completely safe from network drops.
* **SSD Lifespan Protection:** By relying on memory-mapping and eliminating redundant physical window copies, the pipeline drastically reduces read/write cycles, actively protecting your local NVMe/SSD hardware from wear and tear.

### 📊 Comprehensive Execution Benchmarks

| Pipeline Phase | Data Scope / Format | Data Size | Metric / Throughput | Wall Time |
| :--- | :--- | :--- | :--- | :--- |
| **Secure Multi-threaded Download** | Multi-part Raw Archive (`.zip`) | 12.25 GB | 232 MB/s (8 threads) | **03 min 58s** |
| **Parallel Archive Extraction** | Plain-Text File Tree (`.csv`) | 31.04 GB | 116 MB/s (SSD Bound) | **05 min 49s** |
| **Fast C-Engine Ingestion** | Optimized Binary DB (`.zarr`) | 7.27 GB | 4.86 files / second | **11 min 20s** |
| **Local Parallel Packing** | Temporary Local Sync Archive | 7.27 GB | 72.4 MB/s (Stored Mode) | **01 min 38s** |
| **Single-Stream Cloud Sync (Push)** | Google Drive Upload Destination | 7.27 GB | 93.1 MB/s (Buffered I/O) | **01 min 22s** |
| **Cloud Synchronization (Pull)** | Drive-to-Local Resync Engine | 7.27 GB | 39.5 MB/s (Network Bound) | **03 min 04s** |
| **Parallel Scientific Filtering** | Decimated Target Binary Store | 1.82 GB | 14.20 files / second | **03 min 13s** |
| **Lazy Tensor Mapping** | `X_base` / `Y_base` Memmap Arrays | 1.09 GB | **Net RAM Delta: ~7.5 MB** | **00 min 18s** |

👉 **[Run the interactive End-to-End Pipeline in Colab right now!](https://colab.research.google.com/drive/1_QO6B5rM79knfOL3ghE3jFoH533yIVwb?usp=sharing)**

---

## 🛠️ Ecosystem Extensions: CWRU-Plus

If you are expanding your industrial fault diagnosis research beyond the MAFAULDA dataset, check out our companion open-source library: **CWRU-Plus**.

**CWRU-Plus** is a modernized, high-performance data engineering framework specifically designed for the **Case Western Reserve University (CWRU) bearing dataset**.

📦 **PyPI:** `pip install cwru-plus`  
* 🌐 **GitHub Repository:** [Discover CWRU-Plus on GitHub](https://github.com/Razani-Ali/cwru-plus)

## 🤝 Contributing & License

Contributions, bug reports, and feature requests are highly welcome! We built this to accelerate Fault Diagnosis research globally.

This project is open-source and licensed under the **MIT License**.
