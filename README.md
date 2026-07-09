<div align="center">

# 🚀 MAFAULDA-Plus: The Zero-RAM Machinery Dataset Engine

**An enterprise-grade, concurrent, and zero-copy ingestion, filtering, and windowing pipeline for the massive MAFAULDA machinery fault diagnosis dataset.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L1r4yNgGGM-q44tjGChHjc0Un9YB8wqK?usp=sharing)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/mafaulda-plus.svg)](https://badge.fury.io/py/mafaulda-plus)

</div>

---

## 🌟 The Big Data Challenge, Conquered
Following the massive success of our previous library, **CWRU-Plus**, we realized that modern Deep Learning researchers needed an even more powerful engine. While the CWRU dataset is manageable in size, the MAFAULDA dataset is an absolute beast—dozens of gigabytes of raw, multi-channel vibration signals. 

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

# 1 & 2. Download and Extract the massive dataset safely
mafaulda.download(target_path="data/MAFAULDA.zip")
mafaulda.utilities.extract_zip(zip_path="data/MAFAULDA.zip", extract_to="data/MAFAULDA")

# 3. Ingest raw CSVs into a highly compressed Zarr v3 database via multi-processing
mafaulda.ingest(raw_data_dir="data/MAFAULDA", zarr_store_path="data/MAFAULDA.zarr")

# 4. Load a Zero-RAM Memmap tensor (Filtering specific domains instantly!)
X, Y, meta = mafaulda.load(zarr_path="data/MAFAULDA.zarr", target_classes=['normal', 'imbalance'], use_memmap=True)

# 5. Spawn a Native PyTorch DataLoader using our Zero-Copy Virtual Windowing engine
dataloader = mafaulda.get_pytorch_dataloader(X, Y, window_size=1024, step_size=512, class_to_idx={'normal':0, 'imbalance':1})
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

## 🔬 Advanced: Scientific Digital Signal Processing (DSP)

In real-world condition monitoring, naive data loading introduces catastrophic bottlenecks. **MAFAULDA-Plus** addresses two of the most crucial, yet frequently ignored, signal processing pitfalls in machinery fault diagnosis:

### 1. The Tachometer Overfitting Trap 🎯
Many researchers naively feed raw, noisy tachometer pulse signals directly into deep neural networks. Because high-frequency noise corrupts the pulse edges, deep models easily overfit to transient noise spikes rather than the underlying rotational mechanics. Our **`TachometerProcessor`** provides an enterprise-grade, single-responsibility solution: it applies low-pass filtering, robust percentile scaling, and custom rising-edge separation to dynamically calculate the precise, smooth instantaneous RPM vector.

### 2. Piezoelectric Accelerometer Resonance Interference ($23\text{ kHz}$) 🛡️
The physical accelerometers used to record the MAFAULDA dataset exhibit a notorious high-frequency resonance/interference peak centered around **$23\text{ kHz}$**. Feeding this un-attenuated resonance into a CNN or Vision Transformer forces the model to heavily overfit to the sensor's internal structural artifacts rather than the actual rolling element degradation. Our **`AccelerometerFilter`** uses a highly aggressive, causal Elliptic/Butterworth IIR filter to cut off these parasitic frequencies, enforcing strict generalization across cross-domain environments.

Here is how you inject this scientific pipeline concurrently across all CPU cores:

```python
import mafaulda

# 🚀 Step 1: Initialize the advanced Tachometer Pulse Edge-Differentiator
tacho_processor = mafaulda.TachometerProcessor(
    Fs=50000, 
    filter_cutoff=400.0,     # Clean high-frequency noise from pulse streams
    pulses_per_rev=1
)

# 🚀 Step 2: Initialize the causal IIR filter to eliminate the 23kHz sensor resonance
accel_filter = mafaulda.AccelerometerFilter(
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

## ☁️ Google Colab Integration & Real-World Benchmarks

Training on the cloud? MAFAULDA-Plus is fully optimized for Google Colab environments.

While our official Colab execution time and CPU vs. Wall Time benchmark table is currently compiling and will be released in the next update, you don't have to wait to feel the speed. You can experience the multi-threaded extraction, Zarr v3 compression, and zero-RAM PyTorch integration right now.

👉 **[Run the interactive End-to-End Pipeline in Colab right now!](https://colab.research.google.com/drive/1L1r4yNgGGM-q44tjGChHjc0Un9YB8wqK?usp=sharing)**

---

## 🤝 Contributing & License

Contributions, bug reports, and feature requests are highly welcome! We built this to accelerate Fault Diagnosis research globally.

This project is open-source and licensed under the **MIT License**.
