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

def trainTestSplit(dataset,split = 0.7):
    count = int(split * len(dataset))
    return (dataset[:count],dataset[count:])

def getAccuracy(testSet, prediction):
    correct = 0
    genreLabels = {0: "classical", 1: "disco", 2: "hiphop", 3: "metal", 4: "blues", 5: "country"}
    genres = list(genreLabels.values())
    confusionMatrix=[[0 for j in range(6)]for i in range(6)]

    TP = {genre: 0 for genre in genres}     # True Positive
    TN = {genre: 0 for genre in genres}     # True Negative
    FP = {genre: 0 for genre in genres}     # False Positive
    FN = {genre: 0 for genre in genres}     # False Negative

    for x in range(len(testSet)):
        true_label = genreLabels.get(testSet[x][-1])
        predicted_label = genreLabels.get(prediction[x])
        confusionMatrix[prediction[x]][testSet[x][-1]] += 1

        for genre in genres:
            if true_label == genre:
                if predicted_label == genre:
                    TP[genre] += 1
                else:
                    FN[genre] += 1
                    FP[predicted_label] += 1
            elif predicted_label != genre:
                TN[genre] += 1

        print(f"To jest : {true_label}, a to nasze przewidywanie {predicted_label}")
        if testSet[x][-1] == prediction[x]:
            correct += 1

    print("\nMacierz pomyłek:")
    for i in range(6):
        print(confusionMatrix[i])
    print()

    metrics = {}
    for genre in genres:
        accuracy = (TP[genre] + TN[genre]) / (TP[genre] + TN[genre] + FP[genre] + FN[genre])
        sensitivity = TP[genre] / (TP[genre] + FN[genre])
        precision = TP[genre] / (TP[genre] + FP[genre])
        F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
        specificity = TN[genre] / (TN[genre] + FP[genre])

        metrics[genre] = {
            'Accuracy': round(accuracy * 100, 2),
            'Sensitivity-Recall': round(sensitivity * 100, 2),
            'Precision': round(precision * 100, 2),
            'F1': round(F1 * 100, 2),
            'Specificity': round(specificity * 100, 2)
        }

    for genre, metric in metrics.items():
        print(f"{genre.capitalize()} metrics:")
        for key, value in metric.items():
            print(f"{key}: {value}%")
        print("\n")

    return (correct / len(testSet),metrics)