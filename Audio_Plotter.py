import matplotlib.pyplot as plt
import numpy as np

def plot_overlay_waveform(y_input, y_denoised, sr, title="Audio Comparison"):
    
    # Ensure same length
    min_len = min(len(y_input), len(y_denoised))
    y_input = y_input[:min_len]
    y_denoised = y_denoised[:min_len]

    # Time axis
    t = np.arange(min_len) / sr

    # Plot overlay
    plt.figure(figsize=(12, 4))
    plt.plot(t, y_input, alpha=0.6, label="Input (Noisy)", color="red")
    plt.plot(t, y_denoised, alpha=0.8, label="Denoised", color="blue")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
