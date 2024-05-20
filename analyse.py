import pickle
import random as rd

def loadDataset(filename):
    dataset = []
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    return dataset

def splitDataset(dataset, split, trset, teset):
    rd.seed(1)
    rd.shuffle(dataset)

    splitRange = int(split * len(dataset))
    for i in range(splitRange):
        trset.append(dataset[i])
    for j in range(splitRange, len(dataset)):
        teset.append(dataset[j])

def getAccuracy(testSet, prediction):
    correct = 0
    genreLabels = {1: 'classical', 2: 'disco', 3: 'hiphop', 4: 'metal'}
    # True Positive
    TPClassical = 0
    TPDisco = 0
    TPHiphop = 0
    TPMetal = 0

    # True Negative
    TNClassical = 0
    TNDisco = 0
    TNHiphop = 0
    TNMetal = 0

    # False Positive
    FPClassical = 0
    FPDisco = 0
    FPHiphop = 0
    FPMetal = 0

    # False Negative
    FNClassical = 0
    FNDisco = 0
    FNHiphop = 0
    FNMetal = 0
    for x in range(len(testSet)):
        genreName = genreLabels.get(testSet[x][-1])
        predictionName = genreLabels.get(prediction[x])


        if genreName == 'classical':
            if predictionName == 'classical':
                TPClassical += 1
                TNDisco += 1
                TNHiphop += 1
                TNMetal += 1
            if predictionName == 'disco':
                FNClassical += 1
                FPDisco += 1
                TNHiphop += 1
                TNMetal += 1
            if predictionName == 'hiphop':
                FNClassical += 1
                TNDisco += 1
                FPHiphop += 1
                TNMetal += 1
            if predictionName == 'metal':
                FNClassical += 1
                TNDisco += 1
                TNHiphop += 1
                FPMetal += 1
        if genreName == 'disco':
            if predictionName == 'classical':
                FPClassical += 1
                FNDisco += 1
                TNHiphop += 1
                TNMetal += 1
            if predictionName == 'disco':
                TNClassical += 1
                TPDisco += 1
                TNHiphop += 1
                TNMetal += 1
            if predictionName == 'hiphop':
                TNClassical += 1
                FNDisco += 1
                FPHiphop += 1
                TNMetal += 1
            if predictionName == 'metal':
                TNClassical += 1
                FNDisco += 1
                TNHiphop += 1
                FPMetal += 1
        if genreName == 'hiphop':
            if predictionName == 'classical':
                FPClassical += 1
                TNDisco += 1
                FNHiphop += 1
                TNMetal += 1
            if predictionName == 'disco':
                TNClassical += 1
                FPDisco += 1
                FNHiphop += 1
                TNMetal += 1
            if predictionName == 'hiphop':
                TNClassical += 1
                TNDisco += 1
                TPHiphop += 1
                TNMetal += 1
            if predictionName == 'metal':
                TNClassical += 1
                TNDisco += 1
                FNHiphop += 1
                FPMetal += 1
        if genreName == 'metal':
            if predictionName == 'classical':
                FPClassical += 1
                TNDisco += 1
                TNHiphop += 1
                FNMetal += 1
            if predictionName == 'disco':
                TNClassical += 1
                FPDisco += 1
                TNHiphop += 1
                FNMetal += 1
            if predictionName == 'hiphop':
                TNClassical += 1
                TNDisco += 1
                FPHiphop += 1
                FNMetal += 1
            if predictionName == 'metal':
                TNClassical += 1
                TNDisco += 1
                TNHiphop += 1
                TPMetal += 1
        print(f"To jest : {genreName}, a to nasze przewidywanie {predictionName}")
        if testSet[x][-1] == prediction[x]:
            correct += 1


    accuracyClasical = (TPClassical + TNClassical) / (TPClassical + TNClassical + FPClassical + FNClassical)
    sensivityClasical = (TPClassical) / (TPClassical + FNClassical)
    precisionClasical = (TPClassical) / (TPClassical + FPClassical)
    F1Clasical = 2 * (precisionClasical * sensivityClasical) / (precisionClasical + sensivityClasical)
    specificityClasical = (TNClassical) / (TNClassical + FPClassical)

    accuracyDisco = (TPDisco + TNDisco) / (TPDisco + TNDisco + FPDisco + FNDisco)
    sensivityDisco = (TPDisco) / (TPDisco + FNDisco)
    precisionDisco = (TPDisco) / (TPDisco + FPDisco)
    F1Disco = 2 * (precisionDisco * sensivityDisco) / (precisionDisco + sensivityDisco)
    specificityDisco = (TNDisco) / (TNDisco + FPDisco)

    accuracyHiphop = (TPHiphop + TNHiphop) / (TPHiphop + TNHiphop + FPHiphop + FNHiphop)
    sensivityHiphop = (TPHiphop) / (TPHiphop + FNHiphop)
    precisionHiphop = (TPHiphop) / (TPHiphop + FPHiphop)
    F1Hiphop = 2 * (precisionHiphop * sensivityHiphop) / (precisionHiphop + sensivityHiphop)
    specificityHiphop = (TNHiphop) / (TNHiphop + FPHiphop)

    accuracyMetal = (TPMetal + TNMetal) / (TPMetal + TNMetal + FPMetal + FNMetal)
    sensivityMetal = (TPMetal) / (TPMetal + FNMetal)
    precisionMetal = (TPMetal) / (TPMetal + FPMetal)
    F1Metal = 2 * (precisionMetal * sensivityMetal) / (precisionMetal + sensivityMetal)
    specificityMetal = (TNMetal) / (TNMetal + FPMetal)

    print("Accuracy Clasical:", round(accuracyClasical * 10000.0) / 100.0, "%")
    print("Sensivity-Recall Clasical:", round(sensivityClasical * 10000.0) / 100.0, "%")
    print("Precision Clasical:", round(precisionClasical * 10000.0) / 100.0, "%")
    print("F1 Clasical:", round(F1Clasical * 10000.0) / 100.0, "%")
    print("specificity Clasical:", round(specificityClasical * 10000.0) / 100.0, "%")

    print("Accuracy Disco:", round(accuracyDisco * 10000.0) / 100.0, "%")
    print("Sensivity-Recall Disco:", round(sensivityDisco * 10000.0) / 100.0, "%")
    print("Precision Disco:", round(precisionDisco * 10000.0) / 100.0, "%")
    print("F1 Disco:", round(F1Disco * 10000.0) / 100.0, "%")
    print("specificity Disco:", round(specificityDisco * 10000.0) / 100.0, "%")

    print("Accuracy Hiphop:", round(accuracyHiphop * 10000.0) / 100.0, "%")
    print("Sensivity-Recall Hiphop:", round(sensivityHiphop * 10000.0) / 100.0, "%")
    print("Precision Hiphop:", round(precisionHiphop * 10000.0) / 100.0, "%")
    print("F1 Hiphop:", round(F1Hiphop * 10000.0) / 100.0, "%")
    print("specificity Hiphop:", round(specificityHiphop * 10000.0) / 100.0, "%")

    print("Accuracy Metal:", round(accuracyMetal * 10000.0) / 100.0, "%")
    print("Sensivity-Recall Metal:", round(sensivityMetal * 10000.0) / 100.0, "%")
    print("Precision Metal:", round(precisionMetal * 10000.0) / 100.0, "%")
    print("F1 Metal:", round(F1Metal * 10000.0) / 100.0, "%")
    print("specificity Metal:", round(specificityMetal * 10000.0) / 100.0, "%")
    return 1.0 * correct / len(testSet)
