import numpy as np
import scipy.fftpack as fftp

def freq_to_mel[T](f: T) -> T:
    return 1125.0 * np.log(1.0 + f / 700.0)

def mel_to_freq[T](m: T) -> T:
    return 700.0 * (np.exp(m / 1125.0) - 1.0)

def highpass_filter(audio: np.ndarray) -> np.ndarray:
    newAudio = np.empty(len(audio),dtype = float)
    newAudio[0] = audio[0]
    for i in range(1,len(audio)):
        newAudio[i] = audio[i] - 0.97 * audio[i - 1]
    return newAudio

def compute_filterbanks(fftLen: int,numBanks: int,sampleRate: int,lowFreq: int,highFreq: int) -> np.ndarray:
    mels = np.linspace(freq_to_mel(lowFreq),freq_to_mel(highFreq),numBanks + 2)
    bins = np.floor((fftLen + 1) * mel_to_freq(mels) / sampleRate)

    filterbanks = np.zeros((numBanks,fftLen // 2 + 1),dtype = float)
    for bankIndex in range(numBanks):
        for index in range(int(bins[bankIndex]),int(bins[bankIndex + 1])):
            filterbanks[bankIndex,index] = (index - bins[bankIndex]) / (bins[bankIndex + 1] - bins[bankIndex])
        for index in range(int(bins[bankIndex + 1]),int(bins[bankIndex + 2])):
            filterbanks[bankIndex,index] = (bins[bankIndex + 2] - index) / (bins[bankIndex + 2] - bins[bankIndex + 1])
    return filterbanks

def mfcc(audio: np.ndarray,sampleRate: int,numFilterBanks: int = 26,numCepCoeffs: int = 13,fftLen: int = 512,winLen: float = 0.025,winStep: float = 0.01) -> np.ndarray:
    audio = highpass_filter(audio)

    winLenRate = int(winLen * sampleRate)
    winStepRate = int(winStep * sampleRate)

    banks = compute_filterbanks(fftLen,numFilterBanks,sampleRate,0,sampleRate // 2)
    result = np.zeros(((len(audio) + winStepRate - len(audio) % winStepRate) // winStepRate,numCepCoeffs))

    resultI = 0
    for i in range(0,len(audio),winStepRate):
        frame = audio[i:min(i + winLenRate,len(audio))]
        if len(frame) < winLenRate:
            frame = np.append(frame,np.zeros(winLenRate - len(frame)))

        powerSpectrum = np.square(np.abs(np.fft.rfft(frame,fftLen))) / winLenRate

        features = np.dot(powerSpectrum,banks.T)
        features = np.where(features == 0,np.finfo(float).eps,features)

        result[resultI] = fftp.dct(np.log(features),type = 2,norm = "ortho")[:numCepCoeffs]
        resultI += 1
    return result

def main():
    import os
    import pickle
    import traceback
    import scipy.io.wavfile as wav

    directory = "./data"
    with open("./my.dat","wb") as f:
        i = 0
        for folder in os.listdir(directory):
            i += 1
            path = directory + "/" + folder
            for file in os.listdir(path):
                try:
                    (rate,sig) = wav.read(path + "/" + file)
                    mfcc_feat = mfcc(sig,rate)
                    covariance = np.cov(np.matrix.transpose(mfcc_feat))
                    pickle.dump((mfcc_feat.mean(0),covariance,i),f)
                except Exception as e:
                    print("Got an exception:",traceback.format_exc(),'in folder:',folder,'filename:',file)

    #import scipy.io.wavfile as wav
    #(freq,audio) = wav.read("./data/classical/classical.00000.wav")
    #coeffs = mfcc(audio,freq)
    #print(coeffs)

    #audio = audio / max(audio)
    #newAudio = mfcc_emphasize(audio)
    #(figure,plots) = plot.subplots(1,2)
    #figure.suptitle("Audio")
    #plots[0].plot(audio)
    #plots[0].set_ylim([-1,1])
    #plots[1].plot(newAudio)
    #plots[1].set_ylim([-1,1])
    #plot.show()

if __name__ == "__main__":
    main()