from spectral_subtraction import process_audio
from noise_profile_extractor import extract_noise_profile
from hampel_filter import repair_impulses
import soundfile as sf

#file_path = "./noisy_testset_wav/p232_014.wav"
file_path = "subtracted_signal_impulsive_5dB.wav"
output_file_name = "output.wav"
# Pass None to let process_audio extract the noise profile internally

window_ms = 10.0
n_sigma = 5.0
replace_with = 'median'
scale = 'mad'
mad_epsilon = 1e-12

x,sr = sf.read(file_path)

y, mask = repair_impulses(x)

sf.write(output_file_name, y, sr)

process_audio(output_file_name, noise_profile=None, desired_FS=16000)