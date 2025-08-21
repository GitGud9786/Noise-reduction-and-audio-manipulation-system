import os 
import scipy.signal as sig   #Resampling and filtering
import numpy as np           #Spectral subtraction, Mona/Stereo operations
import soundfile as sf       #For reading and writing audio files
import librosa               #For stft
from noise_profile_extractor import extract_noise_profile

#get the current working directory
base_path = os.getcwd()
output_path = os.path.join(base_path,"subtracted_signal.wav")

#Resample input file, return resampled signal in same format
def resample(input_signal, old_sample_rate, new_sample_rate):
    if old_sample_rate == new_sample_rate:
        return input_signal, old_sample_rate
    else:
        resampled_signal = sig.resample_poly(input_signal, new_sample_rate, old_sample_rate)
        return resampled_signal.astype(input_signal.dtype)
    
#Only perform stft on mono audio
def stft(audio, dimensions):
    dimensions = audio.ndim
    if dimensions == 1:
        transform = librosa.stft(audio) #mono audio
        return transform
    else:
       #convert to mono
       audio_mono = librosa.to_mono(audio.T)  # Transpose to ensure correct shape
       transform = librosa.stft(audio_mono)
       return transform

def spectral_subtraction(noise_profile_n, input_signal_n):
    
    N = stft(noise_profile_n, noise_profile_n.ndim)
    # Check if STFT was successful
    if N is None:
        print("Error: STFT failed for noise profile")
        return None
        
    # Magnitude of noise profile
    mN = np.abs(N) 

    Y = stft(input_signal_n, input_signal_n.ndim)
    # Check if STFT was successful
    if Y is None:
        print("Error: STFT failed for input signal")
        return None
        
    # Magnitude of input signal
    mY = np.abs(Y) 
    #Phase as angle of input signal
    pY = np.angle(Y)

    #Creates a complex unit-magnitude matrix holding only phase
    # This is used to reconstruct the signal after spectral subtraction   
    poY = np.exp(1j * pY) 

    # Spectral subtraction
    
    # Mean of noise profile
    noise_mean = np.mean(mN, axis=1, dtype="float64")
    #Reshapes the vector to a column so its shape matches mY for broadcasting in the next step
    noise_mean = noise_mean[:, np.newaxis]
    output_X = mY - noise_mean
    #Clips negative values to zero
    X = np.clip(output_X, a_min=0, a_max=None) 
    #Reconstructs the signal with the phase of the input signal
    X = X * poY 

    #Inverse STFT to get the time-domain signal
    output_signal = librosa.istft(X)
    return output_signal

def process_audio(input_file, noise_profile, desired_FS):
    #Read input signal
    y, fs_y = sf.read(input_file)
    input_dimensions = y.ndim

    if(fs_y != desired_FS):
        y = resample(y,fs_y,desired_FS)
    
    #Read or extract noise profile
    if noise_profile is None:
        # Extract noise profile from input signal
        n = extract_noise_profile(input_file)
        fs_n = fs_y  # Same sample rate as input
    else:
        # Read noise profile from file
        n, fs_n = sf.read(noise_profile)
        if(fs_n != desired_FS):
            n = resample(n, fs_n, desired_FS)
    
    profile_dimensions = n.ndim

    #Check if the noise profile and input signal are mono
    assert profile_dimensions <= 2, "Only mono and stereo files supported for noise profile."
    assert input_dimensions <= 2, "Only mono and stereo files supported for input signal."

    if (profile_dimensions > input_dimensions):
        #make noisy input stereo
        num_channels = profile_dimensions
        y = np.array([y,y], ndmin=num_channels)
        y = np.moveaxis(y, 0, 1)  # Move channels to last dimension
    else:
        #make noise profile stereo
        num_channels = input_dimensions
        n = np.array([n,n], ndmin = num_channels)
        n = np.moveaxis(n, 0, 1)  # to make shape = samples x channels

    #find output for each channel
    #Must transform all mono to stereo for this algorithm to work
    for c in range(num_channels):
        #we work on stereo channels separately
        if num_channels == 1:
            # Mono case
            noise_channel = n
            input_channel = y
        else:
            # Stereo case - extract individual channels
            noise_channel = n[:, c]
            input_channel = y[:, c]
        
        single_channel_output = spectral_subtraction(noise_channel, input_channel)
        if single_channel_output is None:
            print(f"Error processing channel {c}")
            return
        #Initialize output array once
        if (c==0):
         output_x = np.zeros((len(single_channel_output), num_channels))
        output_x[:,c] = single_channel_output

    #Convert all channels to mono if input was mono
    if (num_channels > 1):
        output_x = np.mean(output_x, axis=1)

    #Write output to file
    sf.write(output_path,output_x,desired_FS,format='WAV')
    return
