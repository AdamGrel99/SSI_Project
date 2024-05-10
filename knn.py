import operator
import numpy as np


# odległość Mahalanobisa
def distance(instance1, instance2, k):
    # mm to wektory średnich, czyli mean vector
    # cm to macierze kowariancji, czyli covariance matrices
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    # Odległość ta jest obliczana jako suma 3 tych działań - k (regulacja odległości)
    # 1. iloczyn macierzy odwrotnej cm
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    # 2. kwadrat różnicy między wektorami średnimi, przemnożony przez macierz odwrotną cm2
    distance += (np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1))
    # 3. różnica logarytmów wyznaczników macierzy cm
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

def getNeighbors(trainingset, instance, k):
    distances = []
    for x in trainingset:
        dist = distance(x, instance, k) + distance(instance, x, k)
        distances.append((x[2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def nearestclass(neighbors):
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
