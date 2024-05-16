import knn
import analyse

dataSet = analyse.loadDataset("my.dat")
trainingSet = []
testSet = []

analyse.splitDataset(dataSet, 0.7, trainingSet, testSet)

length = len(testSet)
predictions = []
for x in range(length):
    predict = knn.nearestClass(knn.getNeighbors(trainingSet, testSet[x], 5))
    predictions.append(predict)
accuracy = analyse.getAccuracy(testSet, predictions)
print("Accuracy:", round(accuracy * 10000.0) / 100.0, "%")

