import operator
import numpy as np


def mahalanobisDistance(matrix1, matrix2):
    # cm to macierze kowariancji, czyli covariance matrices
    cm1 = matrix1[1]
    cm2 = matrix2[1]
    return np.trace(np.dot(np.linalg.inv(cm2), cm1))


def getNeighbors(trainingset, matrix, k):
    distances = []
    for x in trainingset:
        dist = mahalanobisDistance(x, matrix) + mahalanobisDistance(matrix, x)
        distances.append((x[2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def nearestClass(neighbors):
    # "1" -> classical, "2" -> disco, "3" -> hiphop, "4" -> metal
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]
