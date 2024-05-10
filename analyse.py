import pickle
import random as rd

def loadDataset(filename):
    rd.seed(1)
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
    rd.shuffle(dataset)
    splitRange = int(split * len(dataset))
    for i in range(len(dataset)):
        if rd.random() < split:
            trset.append(dataset[i])
        else:
            teset.append(dataset[i])

def getAccuracy(testSet, prediction):
    correct = 0
    genreLabels = {1: 'classical', 2: 'disco', 3: 'hiphop', 4: 'metal'}

    for x in range(len(testSet)):
        genreName = genreLabels.get(testSet[x][-1])
        predictionName = genreLabels.get(prediction[x])
        print(f"To jest : {genreName}, a to nasze przewidywanie {predictionName}")
        if testSet[x][-1] == prediction[x]:
            correct += 1
    return 1.0 * correct / len(testSet)