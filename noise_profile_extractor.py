from pyannote.audio import Pipeline
from pyannote.core import Segment
import numpy as np
import soundfile as sf

def extract_noise_profile(input_signal):
  pipeline = Pipeline.from_pretrained(
      "pyannote/voice-activity-detection",
      use_auth_token="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"
  )
  #Load the audio file
  audio, sample_rate = sf.read(input_signal)
  output = pipeline(input_signal)
  
  #Add segments with no speech to noise profile
  no_speech_segments = []
  last_end = 0.0
  for speech in output.get_timeline().support():
      if speech.start > last_end:
          no_speech_segments.append(Segment(last_end,speech.start))
      last_end = speech.end
  
  # Add final segment if there's silence at the end
  audio_duration = len(audio) / sample_rate
  if last_end < audio_duration:
    no_speech_segments.append(Segment(last_end, audio_duration))
  
  #Extract noise profile from quite segments
  noise_samples = []
  for segment in no_speech_segments:
      start = int(segment.start * sample_rate)  # Convert to sample index
      end = int(segment.end * sample_rate)
      if end > start:
       noise_samples.append(audio[start:end])
  
  if noise_samples:
      noise_profile = np.concatenate(noise_samples)
      print(f"Input length: {len(audio)} samples, Noise profile length: {len(noise_profile)} samples")
      return noise_profile
  else:
      #use first 500ms as noise estimate
      noise_profile = audio[:int(0.5*sample_rate)]
      print(f"Input length: {len(audio)} samples, Noise profile length: {len(noise_profile)} samples")
      return noise_profile

if __name__ == "__main__":
  # Example usage
  input_signal = "./noisy_testset_wav/p232_014.wav"
  noise_profile = extract_noise_profile(input_signal)
  
  # Plot noise profile (time domain)
  import matplotlib.pyplot as plt
  
  plt.figure(figsize=(12, 6))
  plt.subplot(2, 1, 1)
  plt.plot(noise_profile)
  plt.title("Noise Profile (Time Domain)")
  plt.xlabel("Sample")
  plt.ylabel("Amplitude")
  
  # Plot noise profile (frequency domain)
  plt.subplot(2, 1, 2)
  from scipy.fft import fft, fftfreq
  N = len(noise_profile)
  yf = fft(noise_profile)
  xf = fftfreq(N, 1/16000)[:N//2]
  plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
  plt.title("Noise Profile (Frequency Domain)")
  plt.xlabel("Frequency (Hz)")
  plt.ylabel("Magnitude")
  plt.tight_layout()
  plt.show()