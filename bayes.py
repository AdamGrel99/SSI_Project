import numpy as np


def gauss(x, mean, std):
    exponent = np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    co = 1 / np.sqrt(2 * np.pi * std ** 2)
    return co * exponent


def matrixToNumber(value):
    return np.sum(value)


def classify(trainset):
    tab = []
    for x in trainset:
        tab.append([x[1], matrixToNumber(x[0])])

    matrix = np.array(tab)
    genreType = np.unique(matrix[:, 0])
    result = {}

    for type in genreType:
        valuesType = matrix[matrix[:, 0] == type][:, 1]
        mean = np.mean(valuesType)
        std = np.std(valuesType)

        result[type] = [mean, std]
    return result


def determineGenre(sample, meanValues):
    prob = []
    meanSample = matrixToNumber(sample[0])
    for key in meanValues:
        meanOfKey = meanValues[key][0]
        stdOfKey = meanValues[key][1]
        prob.append([key, gauss(meanSample, meanOfKey, stdOfKey)])

    maxProb = max(element[1] for element in prob)
    name = 'default'
    for i in prob:
        if i[1] == maxProb:
            name = i[0]

    return name
