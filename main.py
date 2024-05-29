import knn
import analyse
import bayes
import random
from matplotlib import pyplot as plot

random.seed(15)
dataset = analyse.loadDataset("my.dat")
random.shuffle(dataset)
trainingSet,testSet = analyse.trainTestSplit(dataset,0.7)

# knn
results = []
for k in [3,4,5,6,7,8,9,10]:
    print(f"k = {k}:")
    predictions = []
    for x in range(len(testSet)):
        predict = knn.nearestClass(knn.getNeighbors(trainingSet,testSet[x],k))
        predictions.append(predict)
    accuracy,metrics = analyse.getAccuracy(testSet,predictions)
    results.append((k,round(accuracy * 10000.0) / 100.0,metrics))

print("Dokładność dla algorytmu KNN:")
for result in results:
    print(f"  k = {result[0]}: {result[1]}%")

_,knnAccuracyPlot = plot.subplots(1,1)
knnAccuracyPlot.set_title("Dokładność KNN")
knnAccuracyPlot.set_xlabel("Ilość sąsiadów (k)")
knnAccuracyPlot.set_ylabel("Dokładność [%]")
knnAccuracyPlot.bar([result[0] for result in results],[result[1] for result in results])
plot.savefig("./dokładnośćKNN.png")

for genreName in ("classical","disco","hiphop","metal","blues","country"):
    _,knnGenreStats = plot.subplots(1,1)
    knnGenreStats.set_title(f"Metryki KNN - gatunek '{genreName}'")
    knnGenreStats.set_xlabel("Ilość sąsiadów (k)")
    knnGenreStats.set_ylabel("Wartość [%]")

    for metric in ("Accuracy","Sensitivity-Recall","Precision","F1","Specificity"):
        xValues = []
        yValues = []
        for k,_,metrics in results:
            xValues.append(k)
            yValues.append(metrics[genreName][metric])
        knnGenreStats.plot(xValues,yValues,label = metric)

    knnGenreStats.legend()
    plot.savefig(f"./metrykiKNN_{genreName}.png")

print("\n")

# bayes
predictions = []
for x in range(len(testSet)):
    predict = bayes.determineGenre(testSet[x],bayes.classify(trainingSet))
    predictions.append(predict)
accuracy = analyse.getAccuracy(testSet,predictions)
print(f"Dokładność dla Bayesa: {round(accuracy[0] * 10000.0) / 100.0}%")
