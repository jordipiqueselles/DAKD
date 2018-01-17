import os
import sys
import pandas as pd
import multiprocessing as mp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
import sklearn.metrics as metrics
import numpy as np
import logging


def getGoodness(predictor, features, xTrain, yTrain, xTest, yTest):
    predictor.fit(xTrain[:, features], yTrain)
    yPred = predictor.predict(xTest[:, features])
    goodness = metrics.accuracy_score(yTest, yPred)
    return goodness


def getBestFeature(candidateFeatures, selectedFeatures, xTrain, yTrain, xTest, yTest, predictor):
    bestFeature = -1
    bestGoodness = 0

    for feature in candidateFeatures:
        auxSelFeat = list(selectedFeatures)
        auxSelFeat.append(feature)

        goodness = getGoodness(predictor, auxSelFeat, xTrain, yTrain, xTest, yTest)
        logging.info("Feature " + str(feature) + " | Goodness: " + str(goodness))

        if bestGoodness < goodness:
            bestGoodness = goodness
            bestFeature = feature

    return bestFeature, bestGoodness


def getWorstFeature(candidateFeatures, selectedFeatures, xTrain, yTrain, xTest, yTest, predictor):
    worstFeature = -1
    bestGoodness = 0

    for feature in candidateFeatures:
        auxSelFeat = list(selectedFeatures)
        auxSelFeat.remove(feature)

        goodness = getGoodness(predictor, auxSelFeat, xTrain, yTrain, xTest, yTest)
        logging.info("Feature " + str(feature) + " | Goodness: " + str(goodness))

        if bestGoodness < goodness:
            bestGoodness = goodness
            worstFeature = feature

    return worstFeature, bestGoodness


def bidirectionalSearch(xTrain, yTrain, xTest, yTest, nComp, predictor):
    selectedFeatures = set()
    remainingFeat = set(range(nComp))

    while len(selectedFeatures) < len(remainingFeat):
        # adding feature
        candidateFeatures = remainingFeat.difference(selectedFeatures)
        bestFeature, goodness = getBestFeature(candidateFeatures, selectedFeatures, xTrain, yTrain, xTest, yTest, predictor)
        if bestFeature != -1:
            selectedFeatures.add(bestFeature)
        logging.info("Selected features: " + str(selectedFeatures))

        # removing feature
        candidateFeatures = remainingFeat.difference(selectedFeatures)
        worstFeature, goodness = getWorstFeature(candidateFeatures, remainingFeat, xTrain, yTrain, xTest, yTest, predictor)
        if worstFeature != -1:
            remainingFeat.remove(worstFeature)
        logging.info("Remaining features: " + str(remainingFeat))

    return list(selectedFeatures)


def featureSelectPCA(params):
    (predName, predictor, datName, data, listNComp, write) = params
    fileName = 'raw_results/' + predName + '_' + datName + '.txt'
    print(predictor.__class__)

    print(datName)
    X, y = data.load_data()
    logging.info("Num examples " + str(len(X)))

    size = len(X) // 4
    # normalize data
    scale(X, axis=0)

    if write:
        file = open(fileName, 'w')
    else:
        file = None

    for nComp in listNComp:
        logging.info("nComp: " + str(nComp))
        pca = PCA(nComp)
        pca.fit(X)

        shuffledIndices = np.array(range(len(X)))
        np.random.shuffle(shuffledIndices)

        trainingElements = shuffledIndices[:size]
        xTrain = pca.transform(X[trainingElements])
        yTrain = y[trainingElements]


        validationElements = shuffledIndices[size:size*3]
        xVal = pca.transform(X[validationElements])
        yVal = y[validationElements]

        bestFeatures = bidirectionalSearch(xTrain, yTrain, xVal, yVal, nComp, predictor)
        print("\nnComp:", nComp, file=file)
        print("\nnComp:", nComp)
        print("Best features:", bestFeatures, file=file)
        print("Best features:", bestFeatures)

        # Test differences accuracy
        finalTrainingElements = shuffledIndices[:size]
        xFinalTrain = pca.transform(X[finalTrainingElements])
        yFinalTrain = y[finalTrainingElements]
        testingElements = shuffledIndices[size*3:]
        xTest = pca.transform(X[testingElements])
        yTest = y[testingElements]

        predictor.fit(xFinalTrain[:, range(nComp//2)], yFinalTrain)
        testPred = predictor.predict(xTest[:, range(nComp//2)])
        acc = metrics.accuracy_score(yTest, testPred)
        print("Accuracy half pca feat:", acc, file=file)
        print("Accuracy half pca feat:", acc)

        predictor.fit(xFinalTrain[:, bestFeatures], yFinalTrain)
        testPred = predictor.predict(xTest[:, bestFeatures])
        acc = metrics.accuracy_score(yTest, testPred)
        print("Accuracy best pca feat:", acc, file=file)
        print("Accuracy best pca feat:", acc)

        print('\n')
    if write:
        file.close()


class Dataset:
    def __init__(self, path):
        df = pd.read_csv(path)
        self.X = df.iloc[:,:-1].values
        self.y = df.iloc[:,-1].values

    def load_data(self):
        return self.X, self.y


def usage():
    print("Usage:")
    print()
    print("python3 main.py folderDatasets minC maxC predictors [-v] [-p] [-h]")
    print()
    print("folderDatasets -> the folder that contains the datasets in csv format")
    print("minC -> Minimum number of components")
    print("maxC -> Maximum number of components")
    print("predictors -> List of predictors to use. The valid values are lr, nb, kn, dt, rf, all")
    print("-v (optional) -> verbose")
    print("-p (optional) -> parallel")
    print("-h (optional) -> help")
    print()
    print("Examples:")
    print("python3 main.py ./datasets 4 10 all -p")
    print("python3 main.py ./datasets 12 13 ['lr','dt'] -v")


if __name__ == '__main__':
    if '-h' in sys.argv:
        usage()
        exit(0)

    if not 5 <= len(sys.argv) <= 7:
        print("Invalid number of arguments")
        print()
        usage()
        exit(1)

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    if '-v' not in sys.argv:
        logging.disable(logging.INFO)

    if '-p' in sys.argv:
        nCores = mp.cpu_count()
    else:
        nCores = 1

    folderDatasets = sys.argv[1]
    if not os.path.exists(folderDatasets):
        print("The folder", folderDatasets, "doesn't exists")
        exit(1)
    if folderDatasets[-1] != '/':
        folderDatasets += '/'
    listDataFile = {}
    for file in os.listdir(folderDatasets):
        if file[-4:] == '.csv':
            listDataFile[file] = Dataset(folderDatasets + file)

    try:
        minC = int(sys.argv[2])
    except ValueError:
        print("Invalid value for minC", sys.argv[2])
        exit(1)

    try:
        maxC = int(sys.argv[3])
    except ValueError:
        print("Invalid value for maxC", sys.argv[3])
        exit(1)

    allPredictors = {'logisticRegression': LogisticRegression(), 'naiveBayes': GaussianNB(),
                      'kNeighbors': KNeighborsClassifier(), 'decisionTree': DecisionTreeClassifier(),
                      'randomForest': RandomForestClassifier()}
    if 'all' == sys.argv[4]:
        listPredictors = allPredictors
    else:
        argPred = eval(sys.argv[4])
        if type(argPred) is not list:
            print("Invalid type for predictors", sys.argv[4])
            exit(1)
        listPredictors = {}
        for pred in argPred:
            if 'lr' == pred:
                listPredictors['logisticRegression'] = allPredictors['logisticRegression']
            elif 'nb' == pred:
                listPredictors['naiveBayes'] = allPredictors['naiveBayes']
            elif 'kn' == pred:
                listPredictors['kNeighbors'] = allPredictors['kNeighbors']
            elif 'dt' == pred:
                listPredictors['decisionTree'] = allPredictors['decisionTree']
            elif 'rf' == pred:
                listPredictors['randomForest'] = allPredictors['randomForest']
            else:
                print("Invalid value for predictor", pred)
                exit(1)

    if not os.path.exists("./raw_results"):
        os.mkdir("./raw_results")

    allCombinations = ((predName, predictor, datName, data, range(minC*2, maxC*2, 2), True)
                       for (predName, predictor) in listPredictors.items()
                       for (datName, data) in listDataFile.items())

    if nCores == 1:
        list(map(featureSelectPCA, allCombinations))
    else:
        with mp.Pool(nCores) as pool:
            pool.map(featureSelectPCA, allCombinations)
