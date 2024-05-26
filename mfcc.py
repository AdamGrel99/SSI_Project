import os
import sys
import time
import typing
import pickle
import traceback
import numpy as np
from matplotlib import pyplot as plot
import scipy.io.wavfile as wav # type: ignore[import-untyped]
import scipy.fftpack as fftp # type: ignore[import-untyped]

_FM_T = typing.TypeVar("_FM_T",np.ndarray,float)

def freq_to_mel(f: _FM_T,/) -> _FM_T:
    return 1125.0 * np.log(1.0 + f / 700.0)

def mel_to_freq(m: _FM_T,/) -> _FM_T:
    return 700.0 * (np.exp(m / 1125.0) - 1.0)

def compute_filterbanks(fftLen: int,numFilterBanks: int,sampleRate: int,lowFreq: int,highFreq: int) -> np.ndarray:
    mels = np.linspace(freq_to_mel(lowFreq),freq_to_mel(highFreq),numFilterBanks + 2)
    bins = np.floor((float(fftLen) + 1.0) * mel_to_freq(mels) / float(sampleRate))

    filterbanks = np.zeros((numFilterBanks,fftLen // 2 + 1),dtype = float)
    for bankIndex in range(numFilterBanks):
        for index in range(int(bins[bankIndex]),int(bins[bankIndex + 1])):
            filterbanks[bankIndex,index] = (index - bins[bankIndex]) / (bins[bankIndex + 1] - bins[bankIndex])
        for index in range(int(bins[bankIndex + 1]),int(bins[bankIndex + 2])):
            filterbanks[bankIndex,index] = (bins[bankIndex + 2] - index) / (bins[bankIndex + 2] - bins[bankIndex + 1])
    return filterbanks

def highpass_filter_in_place(audio: np.ndarray,/) -> None:
    audio -= 0.97 * np.append(audio[1:],[0.0])

def mfcc(
    audio: np.ndarray,
    sampleRate: int,/,*,
    numFilterBanks: int = 26,
    numCepCoeffs: int = 13,
    fftLen: int = 512,
    winLen: float = 0.025,
    winStep: float = 0.01
) -> np.ndarray:
    audio = audio.astype(float)
    highpass_filter_in_place(audio)

    winLenRate = int(winLen * sampleRate)
    winStepRate = int(winStep * sampleRate)

    banks = compute_filterbanks(fftLen,numFilterBanks,sampleRate,0,sampleRate // 2)
    result = np.zeros(((len(audio) + winStepRate - len(audio) % winStepRate) // winStepRate,numCepCoeffs),dtype = float)

    resultI = 0
    for i in range(0,len(audio),winStepRate):
        frame = audio[i:min(i + winLenRate,len(audio))]
        if len(frame) < winLenRate:
            frame = np.append(frame,np.zeros(winLenRate - len(frame)))

        powerSpectrum = np.square(np.abs(np.fft.rfft(frame,fftLen))) / winLenRate

        features = np.dot(powerSpectrum,banks.T)
        features = np.where(features == 0,np.finfo(float).eps,features)

        result[resultI] = fftp.dct(np.log(features),type=2,norm="ortho")[:numCepCoeffs]
        resultI += 1
    return result

def do_important_stuff_for_me() -> None:
    figure,plot0 = plot.subplots(1,1)
    figure.suptitle("Funkcja trójkątna")
    plot0.set_xlabel("x")
    plot0.set_ylabel("y")
    plot0.plot([0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0])
    plot.savefig("./funkcja_trójkątna.png")

    filePath = "./data/classical/classical.00000.wav"
    sampleRate: int
    samples: np.ndarray
    (sampleRate,samples) = wav.read(filePath)
    samples = samples.astype(float)

    highSamples = samples[:]
    highpass_filter_in_place(highSamples)

    xAxis = np.linspace(0.0,len(samples) / sampleRate,len(samples),dtype = float)
    figure,(plot0,plot1) = plot.subplots(1,2)
    figure.suptitle("Częstotliwość próbkowania: " + str(sampleRate) + " Hz, Plik: " + filePath)
    figure.set_size_inches(14,4)
    plot0.set_title("Próbki")
    plot0.set_xlabel("Czas [s]")
    plot0.set_ylabel("Amplituda")
    plot0.plot(xAxis,samples)
    plot1.set_title("Próbki po filtrze górnoprzepustowym")
    plot1.set_xlabel("Czas [s]")
    plot1.set_ylabel("Amplituda")
    plot1.plot(xAxis,highSamples)
    plot.savefig("./próbki.png")

    print(f"Częstotliwość próbkowania: {sampleRate} Hz")
    print(f"Czas: {round(len(samples) / sampleRate,3)} s")

    winLen = 0.025
    winStep = 0.01
    winLenRate = int(winLen * sampleRate)
    winStepRate = int(winStep * sampleRate)

    print(f"Długość okna w próbkach: {winLenRate}")
    print(f"Długość kroku w próbkach: {winStepRate}")

    fftLen: int = 512
    numFilterBanks: int = 26

    lowFreq = 0
    highFreq = sampleRate // 2
    print(f"Niska częstotliwość: {lowFreq} Hz, wysoka częstotliwość: {highFreq} Hz")

    lowMel = freq_to_mel(lowFreq)
    highMel = freq_to_mel(highFreq)
    print(f"Niska częstotliwość w skali mel: {lowMel}, wysoka częstotliwość w skali mel: {highMel}\n")

    mels = np.linspace(lowMel,highMel,numFilterBanks + 2)
    print(f"Wektor {numFilterBanks + 2} wartości pomiędzy {lowMel} a {highMel}: {mels.tolist()}\n")

    freqs = mel_to_freq(mels)

    naiveFreqs = np.linspace(lowFreq,highFreq,numFilterBanks + 2)
    print(f"Poprzedni wektor zamieniony na częstotliwość: {freqs.tolist()}\n")
    print(f"Częstotliwości {lowFreq} Hz oraz {highFreq} Hz interpolowane bezpośrednio: {naiveFreqs.tolist()}\n")

    figure,plot0 = plot.subplots(1,1)
    plot0.plot(naiveFreqs,"bo",label = "Hz")
    plot0.plot(freqs,"ro",label = "Hz -> Mel -> Hz")
    plot0.legend()
    plot0.set_xlabel("Próbki")
    plot0.set_ylabel("Częstotliwość [Hz]")
    plot0.set_title("Częstotliwości")
    plot.savefig("./częstotliwości.png")

    bins = np.floor((float(fftLen) + 1.0) * freqs / float(sampleRate))
    print(f"Wektor po operacji floor(({float(fftLen)} + 1.0) * freqs / {float(sampleRate)}): {bins.tolist()}\n")

    filterbanks = np.zeros((numFilterBanks,fftLen // 2 + 1),dtype = float)
    for bankIndex in range(numFilterBanks):
        #print(f"indeks: {bankIndex}")

        #print(f"  {int(bins[bankIndex])} -> {int(bins[bankIndex + 1])}")
        for index in range(int(bins[bankIndex]),int(bins[bankIndex + 1])):
            filterbanks[bankIndex,index] = (index - bins[bankIndex]) / (bins[bankIndex + 1] - bins[bankIndex])
            #print(f"    filterbanks[{bankIndex},{index}] = ({index} - {bins[bankIndex]}) / ({bins[bankIndex + 1]} - {bins[bankIndex]}) = {filterbanks[bankIndex,index]}")

        #print(f"  {int(bins[bankIndex + 1])} -> {int(bins[bankIndex + 2])}")
        for index in range(int(bins[bankIndex + 1]),int(bins[bankIndex + 2])):
            filterbanks[bankIndex,index] = (bins[bankIndex + 2] - index) / (bins[bankIndex + 2] - bins[bankIndex + 1])
            #print(f"    filterbanks[{bankIndex},{index}] = ({bins[bankIndex + 2]} - {index}) / ({bins[bankIndex + 2]} - {bins[bankIndex + 1]}) = {filterbanks[bankIndex,index]}")

    #print(filterbanks[0].tolist())
    #print(filterbanks[1].tolist())
    #print(filterbanks[2].tolist())
    #print(filterbanks[3].tolist())
    #print(filterbanks[4].tolist())
    #print(filterbanks[5].tolist())
    #print(filterbanks[6].tolist())
    #print(filterbanks[7].tolist())
    print(f"filterbanks: {filterbanks.tolist()}\n")

    figure,plot0 = plot.subplots(1,1)
    figure.suptitle("Zespół filtrów")
    for row in filterbanks.tolist():
        plot0.plot(row)
    plot.savefig("./filtry.png")

    print(f"filterbanks.T: {filterbanks.T.tolist()}\n")

    numCepCoeffs: int = 13
    resultRows = (len(highSamples) + winStepRate - len(highSamples) % winStepRate) // winStepRate
    print(f"Ilość wierszy macierzy wynikowej: {resultRows}")
    result = np.zeros((resultRows,numCepCoeffs),dtype = float)

    resultI = 0
    for i in range(0,1,winStepRate):
        print(f"Indeks ramki: {i}")
        frame = highSamples[i:min(i + winLenRate,len(highSamples))]
        if len(frame) < winLenRate:
            frame = np.append(frame,np.zeros(winLenRate - len(frame)))

        print(f"  Ramka: {frame}\n")
        powerSpectrum = np.square(np.abs(np.fft.rfft(frame,fftLen))) / winLenRate
        print(f"  Widmowa gęstość mocy: {powerSpectrum}\n")

        features = np.dot(powerSpectrum,filterbanks.T)
        print(f"  Features: {features}\n")
        features = np.where(features == 0,np.finfo(float).eps,features)

        result[resultI] = fftp.dct(np.log(features),type=2,norm="ortho")[:numCepCoeffs]
        print(f"  {result[resultI]}")
        resultI += 1

if __name__ == "__main__":
    if sys.argv.count("-x") == 1:
        do_important_stuff_for_me()
        exit(0)
    print("Tworzenie bazy danych ...")
    startTime = time.perf_counter_ns()
    Directories = ["classical","disco","hiphop","metal","blues","country"]
    with open("./my.dat","wb") as f:
        for i in range(len(Directories)):
            path = "./data/" + Directories[i]
            for file in os.listdir(path):
                try:
                    (rate,sig) = wav.read(path + "/" + file)
                    mfcc_feat = mfcc(sig,rate)
                    covariance = np.cov(mfcc_feat.T)
                    pickle.dump((covariance,i),f)
                except Exception as e:
                    print(f"Wyjątek w folderze '{Directories[i]}' w pliku '{file}': {traceback.format_exc()}")
    endTime = time.perf_counter_ns()
    print(f"Czas wykonania: {(endTime - startTime) / 1e9} s")