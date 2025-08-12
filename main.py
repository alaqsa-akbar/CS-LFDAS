from CompressedSensing import CompressedSensing
from LowPassFilter import LowPassFilter
from DAS import DAS
from utils import plot_signal, plot_signal_3d, rmse, snr, psnr
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract LF-DAS signals.")
    parser.add_argument('--n_channels', type=int, default=20, help='Number of DAS channels')
    parser.add_argument('--time', type=int, default=5, help='Duration of the signal in seconds')
    parser.add_argument('--fs', type=int, default=1000, help='Sampling frequency in Hz')
    parser.add_argument('--noise_std', type=float, default=0.05, help='Standard deviation of additive Gaussian noise')
    parser.add_argument('--frequencies', type=float, nargs='+', default=[0.01, 0.1, 1, 10], help='Frequencies to use in the signal')
    parser.add_argument('--channel_spacing', type=int, default=200, help='Channel spacing in meters')
    parser.add_argument('--velocity_type', choices=['same', 'different', 'variable'], default='same', help='Type of velocity model to use')
    parser.add_argument('--velocity', type=float, default=4000, help='Wave propagation velocity in m/s')
    parser.add_argument('--velocities', type=float, nargs='+', default=[4000, 1000, 3000, 2000], help='List of wave propagation velocities for each frequency')
    parser.add_argument('--chunk_size', type=int, default=50, help='Number of time samples per block (should be divisible by fs * time)')
    parser.add_argument('--overlap', type=float, default=0.5, help='Fraction overlap between consecutive blocks')
    parser.add_argument('--subsample_ratio', type=float, default=0.1, help='Fraction of block elements to measure')
    parser.add_argument('--energy_ratio', type=float, default=0.7, help='Fraction of energy to retain (0 to 1)')
    parser.add_argument('--alpha', type=float, default=5e-4, help='Lasso regularization weight')
    parser.add_argument('--max_iter', type=int, default=600, help='Maximum number of iterations for Lasso')
    parser.add_argument('--cs_cutoff_freq', type=float, default=10, help='Optional pre-filter cutoff (Hz)')
    parser.add_argument('--cs_order', type=int, default=4, help='Order of the Butterworth filter for CSCompress')
    parser.add_argument('--lpf_cutoff_freq', type=float, default=1, help='Low-pass filter cutoff frequency (Hz)')
    parser.add_argument('--lpf_order', type=int, default=4, help='Order of the Butterworth filter for LPFCompress')
    parser.add_argument('--plot_3d', action='store_true', help='Plot signals in 3D if set')
    args = parser.parse_args()

    n_channels = args.n_channels
    time = args.time
    fs = args.fs
    noise_std = args.noise_std
    frequencies = args.frequencies
    channel_spacing = args.channel_spacing
    velocity_type = args.velocity_type
    velocity = args.velocity
    velocities = args.velocities
    if velocity_type == 'different':
        if len(velocities) != len(frequencies):
            raise ValueError("For 'different' velocity type, 'velocities' must have the same length as 'frequencies'")
    chunk_size = args.chunk_size
    overlap = args.overlap
    subsample_ratio = args.subsample_ratio
    energy_ratio = args.energy_ratio
    alpha = args.alpha
    max_iter = args.max_iter
    cs_cutoff_freq = args.cs_cutoff_freq
    cs_order = args.cs_order
    lpf_cutoff_freq = args.lpf_cutoff_freq
    lpf_order = args.lpf_order

    das_signal = DAS(
        n_channels=n_channels,
        time=time,
        fs=fs,
        noise_std=noise_std,
        channel_spacing=channel_spacing,
        frequencies=frequencies,
    )
    if velocity_type == 'same':
        clean_data, data = das_signal.generate_signal(velocity_type=velocity_type, velocity=velocity)
    elif velocity_type == 'different':
        clean_data, data = das_signal.generate_signal(velocity_type=velocity_type, velocities=velocities)
    elif velocity_type == 'variable':
        clean_data, data = das_signal.generate_signal(velocity_type=velocity_type)

    cs_filter = CompressedSensing(
        data=data,
        fs=fs,
        chunk_size=chunk_size,
        overlap=overlap,
        subsample_ratio=subsample_ratio,
        energy_ratio=energy_ratio,
        alpha=alpha,
        max_iter=max_iter,
        cutoff_freq=cs_cutoff_freq,
        order=cs_order
    )
    cs_data = cs_filter.extract_low_freq()

    lpf_filter = LowPassFilter(
        data=data,
        cutoff_freq=lpf_cutoff_freq,
        fs=fs,
        order=lpf_order
    ) 
    lpf_data = lpf_filter.extract_low_freq()

    if n_channels == 1:
        lpf_data = lpf_data.flatten()
        cs_data = cs_data.flatten()

    if not args.plot_3d:
        plot_signal(clean_data, fs, title="Clean Signal")
        cs_filter.plot_original()
        lpf_filter.plot_filtered()
        cs_filter.plot_lf()
    else:
        plot_signal_3d(clean_data, fs, title="Clean Signal 3D")
        cs_filter.plot_original_3d()
        lpf_filter.plot_filtered_3d()
        cs_filter.plot_lf_3d()

    print("RMSE between clean and filtered data:", rmse(clean_data, lpf_data))
    print("RMSE between clean and compressed data:", rmse(clean_data, cs_data))

    print("RMSE between noisy and filtered data:", rmse(data, lpf_data))
    print("RMSE between noisy and compressed data:", rmse(data, cs_data))

    print("SNR of filtered data:", snr(clean_data, lpf_data))
    print("SNR of compressed data:", snr(clean_data, cs_data))

    print("PSNR of filtered data:", psnr(clean_data, lpf_data))
    print("PSNR of compressed data:", psnr(clean_data, cs_data))

if __name__ == "__main__":
    main()
