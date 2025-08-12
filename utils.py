import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fftpack import dctn, idctn
import matplotlib.pyplot as plt


def lpf(
    data: np.ndarray,
    cutoff_freq: float,
    fs: float = 1.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter along the last axis.

    Args:
        data: Input signal (1D or 2D array).
        cutoff_freq: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyq = 0.5 * fs
    norm_cutoff = cutoff_freq / nyq
    if norm_cutoff <= 0 or norm_cutoff >= 1:
        return data
    b, a = butter(order, norm_cutoff, btype='lowpass', analog=False)
    return filtfilt(b, a, data, axis=-1)


def dct2d_flat(vector: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Compute the 2D Discrete Cosine Transform (DCT) and flatten result.

    Args:
        vector: Flattened input array.
        shape: Original 2D shape.

    Returns:
        np.ndarray: Flattened DCT coefficients.
    """
    return dctn(vector.reshape(shape), norm='ortho').ravel()


def idct2d(coeffs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    Compute the inverse 2D Discrete Cosine Transform (DCT).

    Args:
        coeffs: Flattened DCT coefficients.
        shape: Target output shape.

    Returns:
        np.ndarray: Reconstructed 2D signal.
    """
    return idctn(coeffs.reshape(shape), norm='ortho')


def construct_measurement_indices(
    energy_ratio: float,
    n_measurements: int,
    signal_chunk: np.ndarray
) -> np.ndarray:
    """
    Select measurement indices from a 2D block based on energy and randomness.

    70% of indices are chosen based on spectral energy, 30% randomly.

    Args:
        energy_ratio: Ratio of measurements based on energy (0 to 1).
        n_measurements: Total number of measurements to select.
        signal_chunk: 2D signal block.

    Returns:
        np.ndarray: Selected measurement indices (flattened index positions).
    """
    fft2d = np.fft.fft2(signal_chunk)
    energy = np.abs(fft2d)**2
    prob = energy / energy.sum()
    prob_flat = prob.ravel()

    n_energy = int(energy_ratio * n_measurements)
    n_random = n_measurements - n_energy

    energy_idx = np.random.choice(prob_flat.size, size=n_energy, p=prob_flat, replace=False)
    remaining = np.setdiff1d(np.arange(prob_flat.size), energy_idx)
    random_idx = np.random.choice(remaining, size=n_random, replace=False)

    return np.concatenate([energy_idx, random_idx])


def plot_signal(
    signal: np.ndarray,
    fs: float,
    title: str = "Signal"
) -> None:
    """
    Plot a 1D or 2D signal.

    Args:
        signal: Input signal (1D or 2D).
        fs: Sampling frequency in Hz.
        title: Plot title.
    """
    plt.figure()
    if signal.ndim == 1 or signal.shape[0] == 1:
        x_axis = np.arange(signal.shape[-1]) / fs
        plt.plot(x_axis, signal.flatten())
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
    elif signal.ndim == 2:
        im = plt.imshow(
            signal,
            aspect='auto',
            cmap='seismic',
            origin='lower',
            extent=[0, signal.shape[1] / fs, 0, signal.shape[0]]
        )
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Channel")
        plt.colorbar(im)
    else:
        raise ValueError("Signal must be 1D or 2D.")

    plt.tight_layout()
    plt.show()


def plot_signal_3d(
    signal: np.ndarray,
    fs: float,
    title: str = "3D Signal"
) -> None:
    """
    Plot a 2D signal as a 3D surface.

    Args:
        signal: 2D array to plot (channels × time).
        fs: Sampling frequency in Hz.
        title: Plot title.

    Raises:
        ValueError: If the signal is not 2D.
    """
    if signal.ndim != 2:
        raise ValueError("Signal must be 2D for 3D plotting.")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(signal.shape[1]) / fs
    y = np.arange(signal.shape[0])
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, signal, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Channel")
    ax.set_zlabel("Amplitude")

    plt.show()


def rmse(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE) between two signals.

    Args:
        signal1: First signal.
        signal2: Second signal.

    Returns:
        float: RMSE value.

    Raises:
        ValueError: If signal shapes do not match.
    """
    if signal1.shape != signal2.shape:
        raise ValueError("Signals must have the same shape for RMSE calculation.")
    return np.sqrt(np.mean((signal1 - signal2) ** 2))


def bandpass_rmse(
    signal1: np.ndarray,
    signal2: np.ndarray,
    low_freq: float,
    high_freq: float,
    fs: float
) -> float:
    """
    Compute RMSE after applying a bandpass filter to both signals.

    Args:
        signal1: First signal.
        signal2: Second signal.
        low_freq: Lower cutoff frequency in Hz.
        high_freq: Upper cutoff frequency in Hz.
        fs: Sampling frequency in Hz.

    Returns:
        float: Bandpass-filtered RMSE.
    """
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(4, [low, high], btype='band')

    filtered1 = filtfilt(b, a, signal1)
    filtered2 = filtfilt(b, a, signal2)

    return rmse(filtered1, filtered2)


def snr(clean: np.ndarray, recon: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels.

    Args:
        clean: Original clean signal.
        recon: Reconstructed signal.

    Returns:
        float: SNR value in dB.
    """
    signal_power = np.mean(clean**2)
    noise_power = np.mean((clean - recon)**2)
    return 10 * np.log10(signal_power / noise_power)


def psnr(clean: np.ndarray, recon: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in decibels.

    Args:
        clean: Original clean signal.
        recon: Reconstructed signal.

    Returns:
        float: PSNR value in dB.
    """
    mse = np.mean((clean - recon) ** 2)
    if mse == 0:
        return float('inf')
    peak = np.max(np.abs(clean))
    return 20 * np.log10(peak / np.sqrt(mse))
