import numpy as np


class DAS:
    """
    Synthetic Distributed Acoustic Sensing (DAS) signal generator.

    Supports multiple generation modes:
        - Single-channel superposed frequencies
        - Multi-channel with same velocity
        - Multi-channel with different velocities per frequency
        - Multi-channel with per-channel variable frequencies

    Attributes:
        n_channels (int): Number of spatial channels.
        time (float): Duration of the signal in seconds.
        fs (float): Sampling frequency in Hz.
        frequencies (list[float]): Frequencies for signal generation.
        channel_spacing (float): Distance between adjacent channels (meters).
        noise_std (float): Standard deviation of additive Gaussian noise.
    """

    def __init__(
        self,
        n_channels: int,
        time: float,
        fs: float,
        frequencies: list[float],
        channel_spacing: float = 200.0,
        noise_std: float = 0.01
    ) -> None:
        """
        Initialize the DAS signal generator.

        Args:
            n_channels: Number of spatial channels (must be >= 1).
            time: Signal duration in seconds.
            fs: Sampling frequency in Hz.
            frequencies: Frequencies for signal generation (Hz).
            channel_spacing: Distance between channels in meters.
            noise_std: Standard deviation of Gaussian noise.
        """
        if n_channels < 1:
            raise ValueError("At least one channel is required.")
        self.n_channels = n_channels
        self.time = time
        self.fs = fs
        self.frequencies = frequencies
        self.noise_std = noise_std
        self.channel_spacing = channel_spacing

    def generate_signal(
        self,
        velocity_type: str = 'same',
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic DAS signal.

        Args:
            velocity_type: Mode of velocity assignment:
                - 'same': constant velocity across all channels.
                - 'different': velocity varies by frequency.
                - 'variable': frequency changes with channel index.
            **kwargs:
                - For 'same': velocity (float, m/s).
                - For 'different': velocities (list[float], m/s per frequency).
                - For 'variable': frequencies list must match n_channels.

        Returns:
            clean_signal: Noise-free signal, shape (C, T) or (T,) for 1D.
            noisy_signal: Clean signal with added Gaussian noise.

        Raises:
            ValueError: If `velocity_type` is invalid or parameters are inconsistent.
        """
        if self.n_channels == 1:
            return self._generate_1d_signal()

        if velocity_type == 'same':
            return self._generate_signal_same_velocity(
                velocity=kwargs.get('velocity', 0.002)
            )
        elif velocity_type == 'different':
            return self._generate_signal_different_velocity(
                velocities=kwargs.get('velocities', 0.002)
            )
        elif velocity_type == 'variable':
            return self._generate_signal_variable()
        else:
            raise ValueError("velocity_type must be 'same', 'different', or 'variable'")

    def _generate_1d_signal(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a single-channel signal with superposed frequencies.

        Returns:
            clean_signal: Shape (T,).
            noisy_signal: Clean signal plus Gaussian noise, shape (T,).
        """
        t = np.arange(0, self.time, 1 / self.fs)
        clean_signal = np.zeros_like(t)
        for freq in self.frequencies:
            clean_signal += np.sin(2 * np.pi * freq * t)

        noisy_signal = clean_signal + np.random.normal(
            0.0, self.noise_std, size=t.shape
        )
        return clean_signal, noisy_signal

    def _generate_signal_same_velocity(
        self,
        velocity: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a multi-channel signal with the same velocity across all channels.

        Args:
            velocity: Propagation velocity in m/s (0 for no delay).

        Returns:
            clean_signal: Shape (C, T).
            noisy_signal: Clean signal plus Gaussian noise, shape (C, T).
        """
        t = np.arange(0, self.time, 1 / self.fs)
        clean_signal = np.zeros((self.n_channels, len(t)))
        for ch in range(self.n_channels):
            clean_signal_1d = np.zeros_like(t)
            delay = 0 if velocity == 0 else self.channel_spacing * ch / velocity
            for freq in self.frequencies:
                clean_signal_1d += np.sin(2 * np.pi * freq * (t - delay))
            clean_signal[ch, :] = clean_signal_1d

        noisy_signal = clean_signal + np.random.normal(
            0.0, self.noise_std, clean_signal.shape
        )
        return clean_signal, noisy_signal

    def _generate_signal_different_velocity(
        self,
        velocities: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a multi-channel signal with different velocities per frequency.

        Args:
            velocities: List of velocities (m/s), one per frequency.

        Returns:
            clean_signal: Shape (C, T).
            noisy_signal: Clean signal plus Gaussian noise, shape (C, T).
        """
        t = np.arange(0, self.time, 1 / self.fs)
        clean_signal = np.zeros((self.n_channels, len(t)))
        for ch in range(self.n_channels):
            clean_signal_1d = np.zeros_like(t)
            for i, freq in enumerate(self.frequencies):
                delay = 0 if velocities[i] == 0 else self.channel_spacing * ch / velocities[i]
                clean_signal_1d += np.sin(2 * np.pi * freq * (t - delay))
            clean_signal[ch, :] = clean_signal_1d

        noisy_signal = clean_signal + np.random.normal(
            0.0, self.noise_std, clean_signal.shape
        )
        return clean_signal, noisy_signal

    def _generate_signal_variable(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a multi-channel signal with frequency varying per channel.

        Each channel uses a single frequency from the `frequencies` list.

        Returns:
            clean_signal: Shape (C, T).
            noisy_signal: Clean signal plus Gaussian noise, shape (C, T).

        Raises:
            ValueError: If the length of `frequencies` does not match `n_channels`.
        """
        if len(self.frequencies) != self.n_channels:
            raise ValueError("For 'variable' velocity type, 'frequencies' must match 'n_channels'")
        t = np.arange(0, self.time, 1 / self.fs)
        clean_signal = np.zeros((self.n_channels, len(t)))
        for ch in range(self.n_channels):
            clean_signal_1d = np.sin(2 * np.pi * self.frequencies[ch] * t)
            clean_signal[ch, :] = clean_signal_1d

        noisy_signal = clean_signal + np.random.normal(
            0.0, self.noise_std, clean_signal.shape
        )
        return clean_signal, noisy_signal
