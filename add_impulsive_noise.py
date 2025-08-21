import numpy as np
import soundfile as sf
import scipy.signal as sig

# ---------- USER SETTINGS ----------
clean_path = "subtracted_signal.wav"   # 16-kHz mono WAV
output_path = "subtracted_signal_impulsive_5dB.wav"
target_snr_db = 5                   # Desired input SNR in decibels
num_clicks_per_sec = 8              # Impulse rate
click_duration_ms = 2               # Duration of each click
click_amp = 0.95                    # Peak amplitude of each click (before scaling)
# ------------------------------------

# 1. Load clean speech
clean, sr = sf.read(clean_path)
if clean.ndim > 1:                  # convert stereo → mono
    clean = clean.mean(axis=1)

# 2. Generate impulsive noise (random clicks)
N = len(clean)
t = np.arange(N) / sr
click_len = int(sr * click_duration_ms / 1_000)

# Random click start indices
click_starts = np.random.choice(np.arange(N - click_len),
                                size=int(len(t) / sr * num_clicks_per_sec),
                                replace=False)

impulse_noise = np.zeros_like(clean)
for s in click_starts:
    # Raised-cosine (Hann) shaped click to avoid spectral splatter
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(click_len) / (click_len - 1))
    impulse_noise[s:s + click_len] += click_amp * window

# 3. Scale noise to reach the target SNR
clean_power = np.mean(clean ** 2)
noise_power = np.mean(impulse_noise ** 2)
scaling_factor = np.sqrt(clean_power / (noise_power * 10 ** (target_snr_db / 10)))
scaled_noise = impulse_noise * scaling_factor

# 4. Mix clean speech with scaled impulsive noise
noisy = clean + scaled_noise
noisy = np.clip(noisy, -1.0, 1.0)   # prevent clipping

# 5. Save the noisy file
sf.write(output_path, noisy, sr)
print(f"Saved noisy file to {output_path}")

# ---------- OPTIONAL: Verify SNR ----------
def snr_db(clean_sig, noisy_sig):
    noise = noisy_sig - clean_sig
    return 10 * np.log10(np.sum(clean_sig ** 2) / np.sum(noise ** 2))

print("Actual SNR:", snr_db(clean, noisy), "dB")
