import knn
import analyze

dataSet = analyze.loadDataset("my.dat")
trainingSet = []
testSet = []

analyze.splitDataset(dataSet, 0.7, trainingSet, testSet)

length = len(testSet)
predictions = []
for x in range(length):
    predict = knn.nearestclass(knn.getNeighbors(trainingSet, testSet[x], 5))
    predictions.append(predict)
accuracy = analyze.getAccuracy(testSet, predictions)
print("Accuracy:", round(accuracy * 10000.0) / 100.0, "%")

