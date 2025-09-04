import numpy as np
import librosa
import soundfile as sf

def wiener_denoise(
    input_path: str,
    output_path: str,
    noise_duration: float = 0.5,
    n_fft: int = 1024,
    hop_length: int = 512
):

    y_noisy, sr = librosa.load(input_path, sr=None)

    D = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(D), np.angle(D)

    noise_frames = magnitude[:, :int(noise_duration * sr / hop_length)]
    noise_power = np.mean(noise_frames**2, axis=1, keepdims=True)

    signal_power = magnitude**2
    alpha = 12
    H = signal_power / (signal_power + (alpha * noise_power) + 1e-12)  # avoid div by zero
    D_denoised = H * D

    y_denoised = librosa.istft(D_denoised, hop_length=hop_length)

    sf.write(output_path, y_denoised, sr)
    print(f"[INFO] Denoised audio saved at: {output_path} with alpha = {alpha}")

    return y_denoised, sr

y_denoised, sr = wiener_denoise(
    input_path="noisy_fan.wav",
    output_path="denoised_wiener_audio.wav",
    noise_duration=0.5  # use first 0.5s for noise estimation
)
