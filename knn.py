import operator
import numpy as np

def mahalanobisDistance(cm1,cm2):
    # cm to macierze kowariancji, czyli covariance matrices
    return np.trace(np.dot(np.linalg.inv(cm2),cm1))

def getNeighbors(trainingset,instance,k):
    distances = []
    for x in trainingset:
        dist = mahalanobisDistance(x[0],instance[0]) + mahalanobisDistance(instance[0],x[0])
        distances.append((x[1],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]