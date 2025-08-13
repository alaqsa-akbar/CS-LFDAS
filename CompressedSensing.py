import sys
import numpy as np
from scipy.signal import get_window
from sklearn.linear_model import Lasso
from utils import lpf, dct2d_flat, idct2d, construct_measurement_indices, plot_signal, plot_signal_3d
from tqdm import tqdm


class CompressedSensing:
    """
    Implements 2D compressed sensing reconstruction for multi-channel signals
    using block-wise subsampling in the time dimension and sparse recovery in
    the 2D DCT domain via LASSO.

    Attributes:
        data (np.ndarray): Input signal array of shape (C, T), where
            C = number of channels, T = number of time samples.
        fs (float): Sampling frequency in Hz.
        chunk_size (int): Number of time samples per processing block.
        subsample_ratio (float): Fraction of samples in each block to measure.
        energy_ratio (float): Ratio of measurements based on energy (0 to 1).
        alpha (float): LASSO regularization weight.
        max_iter (int): Maximum number of iterations for LASSO solver.
        cutoff (float | None): Optional low-pass filter cutoff frequency in Hz.
        order (int): Order of the Butterworth low-pass filter.
        hop (int): Step size between block starts (derived from overlap).
        filtered (np.ndarray | None): Low-pass filtered data.
        chunks (list[dict]): List of measurement chunks containing:
            - 'start': block start index
            - 'indices': measurement indices (1D)
            - 'y': measured values
        reconstructed (np.ndarray | None): Reconstructed signal array.
    """

    def __init__(
        self,
        data: np.ndarray,
        fs: float,
        chunk_size: int,
        subsample_ratio: float,
        energy_ratio: float,
        alpha: float,
        max_iter: int = 1000,
        cutoff_freq: float | None = None,
        order: int = 4,
        overlap: float = 0.5
    ) -> None:
        """
        Initialize the CompressedSensing object.

        Args:
            data: Input signal, shape (C, T) or (T,) for single-channel.
            fs: Sampling frequency in Hz.
            chunk_size: Number of time samples per block.
            subsample_ratio: Fraction of block samples to retain.
            energy_ratio: Ratio of measurements based on energy (0 to 1).
            alpha: LASSO regularization parameter.
            max_iter: Maximum iterations for LASSO.
            cutoff_freq: Optional pre-filter cutoff frequency in Hz.
            order: Filter order for optional low-pass filter.
            overlap: Fractional overlap between consecutive blocks (0–1).
        """
        self.data = np.array(data)
        if self.data.ndim == 1:
            self.data = self.data[np.newaxis, :]
        self.fs = fs
        self.C, self.T = self.data.shape
        self.chunk_size = chunk_size
        self.hop = int(chunk_size * (1 - overlap))
        self.subsample_ratio = subsample_ratio
        self.n_measurements = int(subsample_ratio * (self.C * chunk_size))
        self.energy_ratio = energy_ratio
        self.alpha = alpha
        self.max_iter = max_iter
        self.cutoff = cutoff_freq
        self.order = order
        self.filtered = None
        self.chunks: list[dict] = []
        self.reconstructed = None

    def extract_low_freq(self) -> np.ndarray:
        """
        Perform compressed sensing reconstruction on low-frequency content.

        Applies optional low-pass filtering, block-based subsampling,
        sparse DCT coefficient recovery via LASSO, and Hann-windowed
        overlap-add reconstruction.

        Returns:
            Reconstructed signal array of shape (C, T).
        """
        data = self.data
        if self.cutoff is not None:
            data = lpf(data, self.cutoff, fs=self.fs, order=self.order)
        self.filtered = data
        self.chunks = []

        for start in range(0, self.T - self.chunk_size + 1, self.hop):
            block = data[:, start:start+self.chunk_size]
            idx = construct_measurement_indices(self.energy_ratio, self.n_measurements, block)
            flat = block.ravel()
            y = flat[idx]
            self.chunks.append({'start': start, 'indices': idx, 'y': y})

        recon = np.zeros_like(self.filtered)
        win = get_window('hann', self.chunk_size)
        win2d = np.tile(win, (self.C, 1))
        win_sum = np.zeros_like(self.filtered)

        for chunk in tqdm(self.chunks, desc="Reconstructing"):
            start = chunk['start']
            idx   = chunk['indices']
            y     = chunk['y']
            N2    = self.C * self.chunk_size

            # Build sensing matrix
            A = np.zeros((idx.size, N2))
            basis = np.zeros(N2)
            for k, i in enumerate(idx):
                basis[i] = 1.0
                A[k, :] = dct2d_flat(basis, (self.C, self.chunk_size))
                basis[i] = 0.0

            # Solve for sparse DCT coefficients
            lasso = Lasso(alpha=self.alpha, fit_intercept=False,
                          max_iter=self.max_iter, tol=1e-5)
            lasso.fit(A, y)
            coeffs = lasso.coef_

            # Reconstruct block
            rec_block = idct2d(coeffs, (self.C, self.chunk_size))
            recon[:, start:start+self.chunk_size] += rec_block * win2d
            win_sum[:, start:start+self.chunk_size] += win2d

        mask = win_sum > 1e-8
        recon[mask] /= win_sum[mask]
        self.reconstructed = recon
        return recon

    def plot_original(self) -> None:
        """Plot the raw multi-channel signal."""
        plot_signal(self.data, self.fs, title="Original Signal")

    def plot_lf(self) -> None:
        """
        Plot the reconstructed low-frequency multi-channel signal.

        Raises:
            ValueError: If reconstruction has not yet been performed.
        """
        if self.reconstructed is None:
            raise ValueError("Call extract_low_freq() before plotting.")
        plot_signal(self.reconstructed, self.fs,
                    title=f"CS Signal (alpha={self.alpha}, subsample_ratio={self.subsample_ratio})")

    def plot_original_3d(self) -> None:
        """Plot the raw multi-channel signal in 3D."""
        plot_signal_3d(self.data, self.fs, title="Original Signal")

    def plot_lf_3d(self) -> None:
        """
        Plot the reconstructed low-frequency multi-channel signal in 3D.

        Raises:
            ValueError: If reconstruction has not yet been performed.
        """
        if self.reconstructed is None:
            raise ValueError("Call extract_low_freq() before plotting.")
        plot_signal_3d(self.reconstructed, self.fs,
                       title=f"CS Signal (alpha={self.alpha}, subsample_ratio={self.subsample_ratio})")

    def get_compressed_size(self) -> int:
        """
        Calculate the total size in bytes of the compressed representation.

        Includes:
            - Python list/dict overhead
            - Stored measurement arrays (`y` and `indices`)
            - Scalar start indices

        Returns:
            Total size of compressed data in bytes.
        """
        size = sys.getsizeof(self.chunks)
        for chunk in self.chunks:
            size += chunk['y'].nbytes
            size += chunk['indices'].nbytes
            size += sys.getsizeof(chunk['start'])
        return size
