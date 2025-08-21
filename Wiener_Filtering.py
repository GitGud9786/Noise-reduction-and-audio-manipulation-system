import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import argparse

class CrowdNoiseFilter:
    def __init__(self):
        pass
    
    def design_voice_bandpass(self, sr, low_freq=80, high_freq=8000):
        """
        Design a band-pass filter optimized for human speech
        """
        nyquist = sr / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        return b, a
    
    def multiband_filter(self, audio, sr):
        """
        Apply different processing to different frequency bands
        """

        low_band = signal.butter(4, [80/sr*2, 300/sr*2], btype='band')
        mid_band = signal.butter(4, [300/sr*2, 3400/sr*2], btype='band')
        high_band = signal.butter(4, [3400/sr*2, 8000/sr*2], btype='band')
        
        # Filter into bands
        low_audio = signal.filtfilt(low_band[0], low_band[1], audio)
        mid_audio = signal.filtfilt(mid_band[0], mid_band[1], audio)
        high_audio = signal.filtfilt(high_band[0], high_band[1], audio)
        
       
        low_gain = 0.3   
        mid_gain = 1.2  
        high_gain = 0.8 
        
        filtered_audio = (low_gain * low_audio + 
                         mid_gain * mid_audio + 
                         high_gain * high_audio)
        
        return filtered_audio
    
    def spectral_gating(self, audio, sr, gate_freq_low=100, gate_freq_high=150):
        """
        Remove specific frequency ranges where crowd noise is prominent
        """
        # FFT
        audio_fft = fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/sr)
        

        gate_mask = np.ones_like(freqs)
        

        gate_indices = ((np.abs(freqs) >= gate_freq_low) & 
                       (np.abs(freqs) <= gate_freq_high))
        
        gate_mask[gate_indices] = 0.1  
        
        # Apply gate and inverse FFT
        filtered_fft = audio_fft * gate_mask
        filtered_audio = np.real(ifft(filtered_fft))
        
        return filtered_audio
    
    def adaptive_bandpass_wiener(self, audio, sr, target_freq_range=(300, 3400)):
        """
        Combine band-pass filtering with Wiener-like spectral subtraction
        """

        low = target_freq_range[0] / (sr/2)
        high = target_freq_range[1] / (sr/2)
        b, a = signal.butter(6, [low, high], btype='band')
        bandpass_audio = signal.filtfilt(b, a, audio)
        

        frame_length = 1024
        hop_length = frame_length // 4
        num_frames = (len(bandpass_audio) - frame_length) // hop_length + 1
        
        filtered_audio = np.zeros_like(bandpass_audio)
        window = np.hanning(frame_length)
        

        noise_frames = int(0.5 * sr / hop_length)
        noise_power_estimate = 0
        
        for i in range(min(noise_frames, num_frames)):
            start_idx = i * hop_length
            end_idx = start_idx + frame_length
            if end_idx <= len(bandpass_audio):
                frame = bandpass_audio[start_idx:end_idx] * window
                frame_power = np.mean(frame ** 2)
                noise_power_estimate += frame_power
        
        noise_power_estimate /= min(noise_frames, num_frames)
        

        for i in range(num_frames):
            start_idx = i * hop_length
            end_idx = start_idx + frame_length
            
            if end_idx > len(bandpass_audio):
                break
                
            frame = bandpass_audio[start_idx:end_idx] * window
            frame_fft = fft(frame)
            frame_power = np.abs(frame_fft) ** 2
            

            frame_energy = np.mean(frame_power)
            
            # Adaptive Wiener-like gain
            if frame_energy > 2 * noise_power_estimate:  # Likely speech
                gain = 0.9  
            else:  # Likely noise/crowd

                wiener_gain = frame_power / (frame_power + 2 * noise_power_estimate)
                gain = np.maximum(wiener_gain, 0.05)
            
            # Apply gain
            if np.isscalar(gain):
                filtered_fft = frame_fft * gain
            else:
                filtered_fft = frame_fft * gain
            

            filtered_frame = np.real(ifft(filtered_fft)) * window
            filtered_audio[start_idx:end_idx] += filtered_frame
        
        return filtered_audio
    
    def process_crowd_noise(self, audio, sr, method='hybrid'):
        """
        Main function to process crowd noise
        
        Methods:
        - 'bandpass': Simple band-pass filter for speech
        - 'multiband': Multi-band processing
        - 'spectral_gate': Remove specific crowd frequency ranges
        - 'hybrid': Combination approach (recommended)
        """
        
        if method == 'bandpass':
            b, a = self.design_voice_bandpass(sr)
            return signal.filtfilt(b, a, audio)
        
        elif method == 'multiband':
            return self.multiband_filter(audio, sr)
        
        elif method == 'spectral_gate':
            return self.spectral_gating(audio, sr)
        
        elif method == 'hybrid':

            audio_step1 = self.multiband_filter(audio, sr)
            

            audio_step2 = self.adaptive_bandpass_wiener(audio_step1, sr)
            
            return audio_step2
        
        else:
            raise ValueError("Method must be 'bandpass', 'multiband', 'spectral_gate', or 'hybrid'")
    
    def process_file(self, input_file, output_file, method='hybrid'):
        """
        Process an audio file to reduce crowd chatter
        """

        try:
            sr, audio = wavfile.read(input_file)
            print(f"Loaded audio: {len(audio)} samples, {sr} Hz sample rate")
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return None, None, None
        

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        print(f"Processing with {method} method...")
        

        filtered_audio = self.process_crowd_noise(audio, sr, method)
        

        max_val = np.max(np.abs(filtered_audio))
        if max_val > 1.0:
            filtered_audio = filtered_audio / max_val
        
        # Save result
        filtered_audio_int = (filtered_audio * 32767).astype(np.int16)
        wavfile.write(output_file, sr, filtered_audio_int)
        print(f"Processed audio saved to: {output_file}")
        
        return audio, filtered_audio, sr


def main():
    parser = argparse.ArgumentParser(description='Crowd Chatter Noise Reduction')
    parser.add_argument('--input', '-i', type=str, help='Input audio file')
    parser.add_argument('--output', '-o', type=str, help='Output audio file')
    parser.add_argument('--method', '-m', 
                       choices=['bandpass', 'multiband', 'spectral_gate', 'hybrid'], 
                       default='hybrid', help='Processing method')
    parser.add_argument('--plot', action='store_true', help='Show frequency analysis plots')
    
    args = parser.parse_args()
    
    if not (args.input and args.output):
        print("Please specify input and output files")
        print("Example: python crowd_filter.py -i noisy.wav -o clean.wav --method hybrid --plot")
        return
    
    
    cnf = CrowdNoiseFilter()
    
    
    original, filtered, sr = cnf.process_file(args.input, args.output, args.method)
    
if __name__ == "__main__":
    main()