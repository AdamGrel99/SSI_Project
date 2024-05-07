import pickle
import random
import operator
import numpy as np

def distance(instance1,instance2,k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2),cm1))
    distance += (np.dot(np.dot((mm2 - mm1).transpose(),np.linalg.inv(cm2)),mm2 - mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

def getNeighbors(trainingset,instance,k):
    distances = []
    for x in range(len(trainingset)):
        dist = distance(trainingset[x],instance,k) + distance(instance,trainingset[x],k)
        distances.append((trainingset[x][2],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def nearestclass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(),key=operator.itemgetter(1),reverse=True)
    return sorter[0][0]

def getAccuracy(testSet,prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    return 1.0 * correct / len(testSet)

def loadDataset(filename,split,trset,teset):
    dataset = []
    with open(filename,'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trset.append(dataset[x])
        else:
            teset.append(dataset[x])

trainingSet = []
testSet = []
loadDataset("my.dat",0.7,trainingSet,testSet)

length = len(testSet)
predictions = []
for x in range(length):
    predictions.append(nearestclass(getNeighbors(trainingSet,testSet[x],5)))
accuracy = getAccuracy(testSet,predictions)
print("accuracy:",round(accuracy * 10000.0) / 100.0,"%")