import numpy as np
from scipy.io import wavfile
import scipy.signal as signal
import os
import matplotlib.pyplot as plt

# ========================= EFFECTS CLASS =========================

def plot_filter_response(original, filtered, sample_rate, title="Filter Response"):
    """
    Plot time domain and frequency domain comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Time domain (first 1000 samples)
    time_orig = np.arange(len(original)) / sample_rate
    time_filt = np.arange(len(filtered)) / sample_rate

    axes[0, 0].plot(time_orig[:1000], original[:1000])
    axes[0, 0].set_title("Original Signal (Time)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].plot(time_filt[:1000], filtered[:1000])
    axes[0, 1].set_title("Filtered Signal (Time)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")

    # Frequency domain
    fft_orig = np.fft.fft(original)
    fft_filt = np.fft.fft(filtered)
    freqs = np.fft.fftfreq(len(original), 1 / sample_rate)
    n = len(freqs) // 2  # positive freqs only

    axes[1, 0].plot(freqs[:n], 20 * np.log10(np.abs(fft_orig[:n]) + 1e-10))
    axes[1, 0].set_title("Original Spectrum")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Magnitude (dB)")
    axes[1, 0].grid(True)

    axes[1, 1].plot(freqs[:n], 20 * np.log10(np.abs(fft_filt[:n]) + 1e-10))
    axes[1, 1].set_title("Filtered Spectrum")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude (dB)")
    axes[1, 1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

class SimpleAudioFilters:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def low_pass_filter(self, audio_data, cutoff_freq=1000):
        """
        Low-pass filter: Removes high frequencies (makes sound muffled)
        Good for: Telephone effect, underwater effect, removing hiss
        """
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, audio_data)
    
    def high_pass_filter(self, audio_data, cutoff_freq=300):
        """
        High-pass filter: Removes low frequencies (makes sound thin/tinny)
        Good for: Removing rumble, making voice sound like intercom
        """
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        return signal.filtfilt(b, a, audio_data)
    
    # ========================= DELAY-BASED EFFECTS =========================
    
    def echo_effect(self, audio_data, delay_ms=500, decay=0.5):
        """
        Echo effect: Adds delayed copy of signal
        Uses simple delay line - fundamental DSP concept
        """
        # Convert delay from milliseconds to samples
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        # Create output array (longer to accommodate echo tail)
        output = np.zeros(len(audio_data) + delay_samples)
        
        # Copy original signal
        output[:len(audio_data)] = audio_data
        
        # Add delayed signal
        output[delay_samples:delay_samples + len(audio_data)] += audio_data * decay
        
        # Trim back to original length if desired, or keep extended
        return output[:len(audio_data)]
    # ========================= FREQUENCY DOMAIN EFFECTS =========================
    
    def pitch_shift(self, audio_data, pitch_factor=1.2):
        """
        Pitch shift using FFT (phase vocoder technique)
        pitch_factor > 1: higher pitch, < 1: lower pitch
        """
        # Simple pitch shift using resampling (changes duration)
        # For time-preserving pitch shift, you'd need phase vocoder
        
        # Resample to change pitch
        original_length = len(audio_data)
        new_length = int(original_length / pitch_factor)
        
        # Use scipy's resampling
        resampled = signal.resample(audio_data, new_length)
        
        # Pad or trim to original length
        if len(resampled) > original_length:
            return resampled[:original_length]
        else:
            padded = np.zeros(original_length)
            padded[:len(resampled)] = resampled
            return padded
    
    # ========================= MODULATION EFFECTS =========================
    
    def tremolo_effect(self, audio_data, rate=5, depth=0.5):
        """
        Tremolo: Amplitude modulation (volume fluctuation)
        Classic guitar effect
        """
        t = np.arange(len(audio_data)) / self.sample_rate
        modulation = 1 + depth * np.sin(2 * np.pi * rate * t)
        return audio_data * modulation
    
    def ring_modulation(self, audio_data, carrier_freq=440):
        """
        Ring modulation: Multiplies signal by sine wave
        Creates metallic, robotic sounds
        """
        t = np.arange(len(audio_data)) / self.sample_rate
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        return audio_data * carrier
    
    # ========================= DISTORTION EFFECTS =========================
    
    def hard_clipping(self, audio_data, threshold=0.1):
        """
        Hard clipping distortion: Sharp cutoff at threshold
        Creates harsh, digital distortion
        """
        return np.clip(audio_data, -threshold, threshold)
    
    # ========================= SPECIALIZED FILTERS =========================
    
    def telephone_filter(self, audio_data):
        """
        Telephone effect: Bandpass + distortion + noise
        Combines multiple techniques
        """
        # Narrow bandpass (300-3400 Hz like old phones)
        filtered = self.low_pass_filter(
            self.high_pass_filter(audio_data, 300), 3400)
        
        # Add slight distortion
        distorted = np.tanh(filtered * 2) * 0.7
        
        # Add noise
        noise = np.random.normal(0, 0.01, len(distorted))
        
        return distorted + noise


# ========================= HELPER =========================
def _to_float32(x):
    """Convert PCM or float to float32 in [-1, 1]."""
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    if x.dtype == np.int32:
        return x.astype(np.float32) / 2147483648.0
    if x.dtype == np.uint8:
        return (x.astype(np.float32) - 128.0) / 128.0
    return x.astype(np.float32)

def _from_float32(x_float, dtype=np.int16):
    """Convert float32 in [-1, 1] to desired PCM dtype (default int16)."""
    x = np.clip(x_float, -1.0, 1.0)
    if dtype == np.int16:
        return (x * 32767.0).astype(np.int16)
    if dtype == np.int32:
        return (x * 2147483647.0).astype(np.int32)
    if dtype == np.uint8:
        return (np.round((x * 127.0) + 128.0)).astype(np.uint8)
    return x.astype(dtype)

def _apply_channelwise(effect_fn, audio_float):
    """Apply effect to mono or each channel of a stereo/multi-channel signal."""
    if audio_float.ndim == 1:
        return effect_fn(audio_float)
    # For multi-channel: process each channel independently
    processed = np.empty_like(audio_float)
    for c in range(audio_float.shape[1]):
        processed[:, c] = effect_fn(audio_float[:, c])
    return processed

# ========================= MAIN (ONE EFFECT ONLY) =========================
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_WAV = os.path.join(SCRIPT_DIR, "input_audio.wav")
    OUTPUT_WAV = os.path.join(SCRIPT_DIR, "output_filtered.wav") # <- set your output file name
    
    

    sample_rate, audio = wavfile.read(INPUT_WAV)
    original_dtype = audio.dtype

    audio_f32 = _to_float32(audio)
    fx = SimpleAudioFilters(sample_rate)

    # ---- CHOOSE EXACTLY ONE EFFECT (no ifs) ----
    # Example: low-pass @ 1200 Hz. Change this ONE line to switch effects.
    print("What type of filter do you want?\n" \
    "Echo Effect (0)\n" \
    "Pitch Shift (1)\n" \
    "Tremolo Effect (2)\n" \
    "Ring Modulation (3)\n" \
    "Hard Clipping Distortion (4)\n" \
    "Telephone Filter (5)\n")


    answer = int(input("Enter a number (0-5): ").strip())
    print
    chosen_effect = None
    if answer == 0:
        delay = int(input("What should be the delay in millisecons?"))
        chosen_effect = lambda x: fx.echo_effect(x, delay_ms=delay)
    elif answer == 1:
        factor = float(input("What should be the pitch factor?"))
        chosen_effect = lambda x: fx.pitch_shift(x, pitch_factor=factor)
    elif answer == 2:
        rate = int(input("What should be the rate?"))
        depth = float(input("What should be the depth?"))
        chosen_effect = lambda x: fx.tremolo_effect(x, rate=rate, depth=depth)
    elif answer == 3:
        freq = int(input("What should be the carrier frequency?"))
        chosen_effect = lambda x: fx.ring_modulation(x, carrier_freq=freq)
    elif answer == 4:
        threshold = float(input("What should be the distortion threshold?"))
        chosen_effect = lambda x: fx.hard_clipping(x, threshold=threshold)  
    elif answer == 5:
        chosen_effect = lambda x: fx.telephone_filter(x)  

    if chosen_effect is None:
        raise ValueError("Invalid choice. Please run again and enter a number 0–5.")

    processed = _apply_channelwise(chosen_effect, audio_f32)

    # Optional safety: peak-normalize if anything exceeded 1.0 during processing
    peak = np.max(np.abs(processed))
    if peak > 1.0:
        processed = processed / peak * 0.99

    # Write out (default to int16 for broad compatibility)
    wavfile.write(OUTPUT_WAV, sample_rate, _from_float32(processed, dtype=np.int16))
    print(f"Done. Wrote: {OUTPUT_WAV}")
    plot_filter_response(audio_f32 if audio_f32.ndim == 1 else audio_f32[:,0],
                     processed if processed.ndim == 1 else processed[:,0],
                     sample_rate,
                     title=f"Effect {answer} Response")