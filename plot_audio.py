import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 1. Load the audio
y, sr = librosa.load("voice.wav", sr=None, mono=True)
print("Sample rate:", sr, "Hz")
print("Number of samples:", len(y))
print("Duration:", len(y) / sr, "s")

# 2. Time axis for the waveform
t = np.arange(len(y)) / sr

# 3. STFT spectrogram
D = librosa.stft(y, n_fft=2048, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
print("STFT shape (frequency bins, time frames):", D.shape)

# 4. Mel spectrogram
M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
M_db = librosa.power_to_db(M, ref=np.max)
print("Mel shape (mel bands, time frames):", M.shape)

# 5. Plot
fig, ax = plt.subplots(3, 1, figsize=(10, 9))

ax[0].plot(t, y)
ax[0].set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")

librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="hz", ax=ax[1])
ax[1].set(title="STFT spectrogram (dB)")

librosa.display.specshow(M_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax[2])
ax[2].set(title="Mel spectrogram (dB)")

plt.tight_layout()
plt.savefig("spectrograms.png", dpi=150)
plt.show()