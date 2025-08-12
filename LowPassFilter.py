import numpy as np
from utils import lpf, plot_signal, plot_signal_3d


class LowPassFilter:
    """
    Applies a low-pass Butterworth filter to single- or multi-channel signals.

    Attributes:
        data (np.ndarray): Input signal array of shape (C, T) or (T,).
        cutoff_freq (float): Filter cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the Butterworth filter.
        filtered (np.ndarray | None): Filtered signal after processing.
    """

    def __init__(
        self,
        data: np.ndarray,
        cutoff_freq: float,
        fs: float,
        order: int = 4
    ) -> None:
        """
        Initialize the LowPassFilter object.

        Args:
            data: Input signal array, shape (C, T) or (T,) for single-channel.
            cutoff_freq: Filter cutoff frequency in Hz.
            fs: Sampling frequency in Hz.
            order: Butterworth filter order.
        """
        self.data = np.array(data)
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.order = order
        self.filtered = None

    def extract_low_freq(self) -> np.ndarray:
        """
        Apply the low-pass filter to the input signal.

        Returns:
            Filtered signal array of the same shape as input.
        """
        self.filtered = lpf(
            self.data,
            cutoff_freq=self.cutoff_freq,
            fs=self.fs,
            order=self.order
        )
        return self.filtered

    def plot_filtered(self) -> None:
        """
        Plot the filtered signal in 2D.

        Raises:
            ValueError: If filtering has not yet been performed.
        """
        if self.filtered is None:
            raise ValueError("Call extract_low_freq() before plotting.")
        plot_signal(self.filtered, self.fs, title="Filtered Signal")

    def plot_filtered_3d(self) -> None:
        """
        Plot the filtered signal in 3D.

        Raises:
            ValueError: If filtering has not yet been performed.
        """
        if self.filtered is None:
            raise ValueError("Call extract_low_freq() before plotting.")
        plot_signal_3d(self.filtered, self.fs, title="Filtered Signal 3D")
