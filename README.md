# Compressed Sensing LF-DAS

## Introduction
Low-frequency Distributed Acoustic Sensing (LF-DAS) signals play a critical role in in-well sensing applications, including thermal transients, strain changes, flow regime characterization, and event detection. However, conventional fixed low-pass filters often prove suboptimal for extracting these signals. The optimal cutoff frequency can vary significantly along the wellbore—particularly in production wells—due to spectral non-stationarity, resulting in trade-offs between signal preservation and noise rejection. Furthermore, different physical phenomena in the wellbore generate distinct low-frequency DAS signatures with unique frequency content, often within non-overlapping spectral bins. This work introduces and evaluates a Compressive Sensing (CS)-based methodology designed to adaptively extract LF-DAS signals without requiring a predefined low cutoff frequency, enabling improved signal recovery and noise suppression in dynamic wellbore environments.

## Methods/Procedures/Process: 
The approach leverages the principle that LF-DAS signals, often representing slower physical processes, exhibit sparsity in transform domains like the Discrete Cosine Transform (DCT). Following initial pre-processing, a conservatively high-cutoff filter is applied primarily to remove high-frequency content well above the typical LF-DAS band of interest. The core of the method then employs 2D CS reconstruction. By randomly subsampling the pre-filtered data and solving an L1-minimization problem, the algorithm reconstructs the signal components that best explain the measurements under the sparsity constraint. This inherently favors the dominant, sparse, low-frequency components without enforcing a sharp, predefined frequency boundary, allowing the effective cutoff to be data-driven and spatially variable.

---

## 📁 Files

### `LowPassFilter.py`

Implements a Butterworth low-pass filter to extract LF-DAS signals.

**Parameters:**
- `data`: Data to be filtered (`np.ndarray` of shape `(C, T)` or `(T,)`)
- `cutoff_freq`: Cutoff frequency in Hz
- `fs`: Sampling frequency in Hz
- `order`: Order of the Butterworth filter (default: `4`)

---

### `CompressedSensing.py`

Implements a compressed sensing pipeline. Data is pre-filtered using a Butterworth low-pass filter (optional), chunked into overlapping windows, and randomly subsampled. Reconstruction is done via Lasso regression in the 2D DCT domain.

**Parameters:**
- `data`: Data to be compressed (`np.ndarray` of shape `(C, T)`)
- `fs`: Sampling frequency in Hz
- `chunk_size`: Number of time samples per block
- `subsample_ratio`: Fraction of elements in each block to measure
- `energy_ratio`: Fraction of energy to retain (0 to 1)
- `alpha`: Lasso regularization weight
- `max_iter`: Maximum number of iterations for Lasso
- `cutoff_freq`: Optional pre-filter cutoff frequency (Hz)
- `order`: Order of the Butterworth filter for pre-filtering
- `overlap`: Fractional overlap between consecutive blocks

---

### `DAS.py`

Simulates Distributed Acoustic Sensing (DAS) signal data. Four signal modes are supported:
- Single-channel signals with superposed frequencies
- Multi-channel signals with the same velocity across channels
- Multi-channel signals with different velocities per frequency
- Multi-channel signals with variable frequencies per channel

**Parameters:**
- `n_channels`: Number of spatial channels
- `time`: Signal duration in seconds
- `fs`: Sampling frequency in Hz
- `frequencies`: List of frequencies for signal generation
- `channel_spacing`: Distance between adjacent channels in meters
- `noise_std`: Standard deviation of additive Gaussian noise

---

### `utils.py`

Provides utility functions shared across modules. Includes low-pass filtering, DCT operations, measurement index construction, visualization helpers, and performance metrics (RMSE, SNR, PSNR).

---

### `main.py`

The main entry point. Use this script to simulate DAS data, compress it using CS or LPF methods, and evaluate the results with performance metrics.

To run with default settings:
```bash
python main.py
```

To customize the run, you may pass optional arguments such as:
```bash
python main.py --frequencies 0.01 0.1 1 10 --time 10 --fs 1000 --noise_std 0.1 --n_channels 20 --subsample_ratio 0.1 --alpha 5e-4 --lpf_cutoff_freq 1 --max_iter 600 --chunk_size 50
```

---

## ⚙️ Command-Line Arguments

| Argument                | Description                                                        | Default         |
|-------------------------|--------------------------------------------------------------------|-----------------|
| `--n_channels`          | Number of DAS channels                                             | `20`            |
| `--time`                | Duration of the signal in seconds                                  | `5`             |
| `--fs`                  | Sampling frequency in Hz                                           | `1000`          |
| `--noise_std`           | Standard deviation of Gaussian noise                               | `0.05`          |
| `--frequencies`         | Frequencies in the signal (space-separated list)                   | `0.01 0.1 1 10` |
| `--channel_spacing`     | Channel spacing in meters                                          | `200`           |
| `--velocity_type`       | Velocity model: 'same', 'different', or 'variable'                | `same`          |
| `--velocity`            | Wave propagation velocity in m/s (for 'same' type)                | `4000`          |
| `--velocities`          | List of velocities for each frequency (for 'different' type)      | `4000 1000 3000 2000` |
| `--chunk_size`          | Number of samples per compressed block                             | `50`            |
| `--overlap`             | Fractional overlap between blocks (0 to 1)                         | `0.5`           |
| `--subsample_ratio`     | Fraction of measurements to keep in CS                             | `0.1`           |
| `--energy_ratio`        | Fraction of energe to retain (0 to 1)                              | `0.7`           |
| `--alpha`               | Lasso regularization strength                                      | `5e-4`          |
| `--max_iter`            | Max number of iterations for Lasso                                 | `600`           |
| `--cs_cutoff_freq`      | Pre-filter cutoff frequency for CS (Hz)                            | `10`            |
| `--cs_order`            | Filter order for CS pre-filter                                     | `4`             |
| `--lpf_cutoff_freq`     | LPF cutoff frequency (Hz)                                          | `1`             |
| `--lpf_order`           | LPF filter order                                                   | `4`             |
| `--plot_3d`             | Enable 3D plotting of signals (flag)                              | `False`         |

---

## ✅ Example Use Cases

- Compare LPF and CS-based low filter data extraction methods on clean or noisy DAS data
- Simulate different signal profiles with varying velocities to evaluate robustness
- Visualize compressed vs. original signal performance in both time and frequency domains
- Assess reconstruction quality using RMSE, SNR, and PSNR metrics

---

## 🧪 Notes

- Ensure `chunk_size` is compatible with the total signal length (`fs * time`) for proper segmentation
- Always set cutoff frequencies below Nyquist (`fs/2`) to avoid aliasing
- For rapid prototyping, start with low `fs`, few `n_channels`, and short `time` duration
- The compressed sensing method adaptively determines effective cutoff frequencies based on signal content
- Performance metrics are computed comparing both methods against clean and noisy reference signals

---