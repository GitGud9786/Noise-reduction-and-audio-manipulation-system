import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

class OldSchoolRadioFilter:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        
    def apply_bandpass_filter(self, audio_data, low_freq=300, high_freq=3400):
        """
        Apply bandpass filter to simulate limited frequency response of old radios
        Old radios typically had frequency response between 300Hz - 3400Hz
        """
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        return filtered_audio
    
    def add_tube_distortion(self, audio_data, gain=2.0, threshold=0.7):
        """
        Simulate tube amplifier distortion characteristic of old radios
        """
        # Amplify signal
        amplified = audio_data * gain
        
        # Apply soft clipping (tube-like distortion)
        distorted = np.tanh(amplified)
        
        # Add some harmonic distortion
        distorted = distorted + 0.1 * np.sin(2 * np.pi * amplified)
        
        return distorted
    
    def add_crackle_noise(self, audio_data, intensity=0.02):
        """
        Add crackle noise to simulate old radio static
        """
        # Generate random impulses for crackling
        crackle_rate = 0.001  # Probability of crackle per sample
        crackles = np.random.random(len(audio_data)) < crackle_rate
        crackle_noise = crackles * np.random.normal(0, intensity, len(audio_data))
        
        return audio_data + crackle_noise
    
    def add_hum(self, audio_data, hum_freq=50, intensity=0.01):
        """
        Add 50Hz hum typical of old electrical equipment
        """
        t = np.arange(len(audio_data)) / self.sample_rate
        hum = intensity * np.sin(2 * np.pi * hum_freq * t)
        
        return audio_data + hum
    
    def apply_amplitude_modulation(self, audio_data, mod_freq=0.5, mod_depth=0.1):
        """
        Apply slight amplitude modulation to simulate power supply variations
        """
        t = np.arange(len(audio_data)) / self.sample_rate
        modulation = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
        
        return audio_data * modulation
    
    def compress_dynamic_range(self, audio_data, ratio=4.0, threshold=0.3):
        """
        Apply compression to simulate limited dynamic range of old radios
        """
        # Simple compression algorithm
        compressed = np.copy(audio_data)
        
        # Find samples above threshold
        above_threshold = np.abs(compressed) > threshold
        
        # Apply compression to samples above threshold
        compressed[above_threshold] = threshold + (compressed[above_threshold] - threshold) / ratio
        
        return compressed
    
    def apply_radio_filter(self, audio_data, 
                          bandpass_params={'low_freq': 300, 'high_freq': 3400},
                          distortion_params={'gain': 1.8, 'threshold': 0.7},
                          noise_params={'crackle_intensity': 0.015, 'hum_intensity': 0.008},
                          modulation_params={'mod_freq': 0.3, 'mod_depth': 0.05},
                          compression_params={'ratio': 3.0, 'threshold': 0.4}):
        """
        Apply complete old-school radio filter chain
        """
        print("Applying old-school radio filter...")
        
        # Step 1: Bandpass filter (most important for radio sound)
        print("1. Applying bandpass filter...")
        filtered_audio = self.apply_bandpass_filter(audio_data, **bandpass_params)
        
        # Step 2: Add tube distortion
        print("2. Adding tube distortion...")
        filtered_audio = self.add_tube_distortion(filtered_audio, **distortion_params)
        
        # Step 3: Compress dynamic range
        print("3. Applying compression...")
        filtered_audio = self.compress_dynamic_range(filtered_audio, **compression_params)
        
        # Step 4: Add amplitude modulation
        print("4. Adding amplitude modulation...")
        filtered_audio = self.apply_amplitude_modulation(filtered_audio, **modulation_params)
        
        # Step 5: Add electrical hum
        print("5. Adding electrical hum...")
        filtered_audio = self.add_hum(filtered_audio, 50, noise_params['hum_intensity'])
        
        # Step 6: Add crackle noise
        print("6. Adding crackle noise...")
        filtered_audio = self.add_crackle_noise(filtered_audio, noise_params['crackle_intensity'])
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(filtered_audio))
        if max_val > 1.0:
            filtered_audio = filtered_audio / max_val * 0.95
        
        print("Radio filter applied successfully!")
        return filtered_audio
    
    def plot_frequency_response(self, original_audio, filtered_audio):
        """
        Plot frequency domain comparison between original and filtered audio
        """
        # Compute FFT
        fft_original = np.fft.fft(original_audio)
        fft_filtered = np.fft.fft(filtered_audio)
        
        # Frequency bins
        freqs = np.fft.fftfreq(len(original_audio), 1/self.sample_rate)
        
        # Plot magnitude spectrum (first half only due to symmetry)
        n = len(freqs) // 2
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(freqs[:n], 20 * np.log10(np.abs(fft_original[:n]) + 1e-10))
        plt.title('Original Audio - Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(freqs[:n], 20 * np.log10(np.abs(fft_filtered[:n]) + 1e-10))
        plt.title('Radio Filtered Audio - Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def process_audio_file(input_file, output_file, plot_spectrum=True):
    """
    Main function to process an audio file with radio filter
    """
    try:
        # Read audio file
        print(f"Reading audio file: {input_file}")
        sample_rate, audio_data = wavfile.read(input_file)
        
        # Convert to float and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        # Handle stereo audio (convert to mono for simplicity)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
        
        # Create filter instance
        radio_filter = OldSchoolRadioFilter(sample_rate)
        
        # Apply radio filter
        filtered_audio = radio_filter.apply_radio_filter(audio_data)
        
        # Plot frequency response comparison
        if plot_spectrum:
            radio_filter.plot_frequency_response(audio_data, filtered_audio)
        
        # Convert back to int16 for saving
        filtered_audio_int16 = (filtered_audio * 32767).astype(np.int16)
        
        # Save filtered audio
        wavfile.write(output_file, sample_rate, filtered_audio_int16)
        print(f"Filtered audio saved to: {output_file}")
        
        return filtered_audio, sample_rate
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        return None, None

# Example usage
if __name__ == "__main__":
    # Example file paths - modify these for your actual files
    input_file = "input_audio.wav"  # Your input audio file
    output_file = "radio_filtered_audio.wav"  # Output file
    
    # Process the audio file
    filtered_audio, sample_rate = process_audio_file(input_file, output_file, plot_spectrum=True)
    
    if filtered_audio is not None:
        print("\n" + "="*50)
        print("OLD SCHOOL RADIO FILTER APPLIED SUCCESSFULLY!")
        print("="*50)
        print("\nFilter characteristics applied:")
        print("✓ Bandpass filter (300Hz - 3400Hz)")
        print("✓ Tube amplifier distortion")
        print("✓ Dynamic range compression")
        print("✓ Amplitude modulation")
        print("✓ 50Hz electrical hum")
        print("✓ Crackle noise")
        print(f"\nOutput saved to: {output_file}")
    else:
        print("Failed to process audio file. Please check your input file path and format.")
        
    # You can also create a custom filter with different parameters
    print("\n" + "-"*40)
    print("Custom filter example:")
    print("-"*40)
    
    # Example with custom parameters for a more aggressive radio effect
    custom_params = {
        'bandpass_params': {'low_freq': 400, 'high_freq': 2800},  # Even more limited bandwidth
        'distortion_params': {'gain': 2.5, 'threshold': 0.6},    # More distortion
        'noise_params': {'crackle_intensity': 0.03, 'hum_intensity': 0.015},  # More noise
        'modulation_params': {'mod_freq': 0.8, 'mod_depth': 0.08},  # More modulation
        'compression_params': {'ratio': 5.0, 'threshold': 0.3}   # More compression
    }
    
    print("To use custom parameters, modify the 'custom_params' dictionary above")
    print("and call: radio_filter.apply_radio_filter(audio_data, **custom_params)")