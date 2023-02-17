# classifier.py
# Lin Li/26-dec-2021
import numpy as np
import random


def featureIndexToSplitOn(features, data, target):
    splitIndex = [gini(featureIndex, data, target) for featureIndex in range(len(features))]
    minIndex = splitIndex.index(min(splitIndex))

    return minIndex


def gini(featureIndex, data, target):
    one, zero = [], []
    targetCounter = [[0] * 4, [0] * 4]

    for i in list(zip(data, target)):
        (one if int(i[0][featureIndex]) else zero).append(i)
        targetCounter[int(i[0][featureIndex])][int(i[1])] += 1

    si = [len(one), len(zero)]
    s = sum(si)
    giniImpurity = 0

    for i in range(2):
        gi = sum([j ** 2 for j in targetCounter[i]])
        gi = 1 - (gi / (s ** 2))
        giniImpurity += (si[i] / s) * gi

    return giniImpurity


def removeFeatureFromData(features, feature_index, data):
    for i in range(2):
        for j in range(len(data[i])):
            data[i][j] = data[i][j][:feature_index] + data[i][j][feature_index + 1:]

    features.pop(feature_index)

    return features, data[0], data[1]


class DecisionTree:
    def __init__(self):
        self.head = None

    def fit(self, data, target, node=None, features=None):
        # check if node is the root node
        if node is None:
            self.head = DecisionNode(target, None)
            node = self.head
            features = list(range(len(data[0])))

        # if there are no example in this branch then make it a leaf with its parents plurality value for class
        if len(data) == 0:
            return LeafNode(node.parent.pluralityValue())

        # if there are no more features to split on
        elif len(features) == 0:
            return LeafNode(node.pluralityValue())

        # if all the example have the same classification then make leaf with that classification
        elif len(set(target)) == 1:
            return LeafNode(target[0])

        else:
            splitOnIndex = featureIndexToSplitOn(features, data, target)
            rightData, rightTarget, leftData, leftTarget = [], [], [], []

            node.featureIndex = features[splitOnIndex]

            # go through data and split on features --> binary split
            for i in list(zip(data, target)):
                if int(i[0][splitOnIndex]):
                    rightData.append(i[0])
                    rightTarget.append(i[1])
                else:
                    leftData.append(i[0])
                    leftTarget.append(i[1])

            newFeatures, newRightData, newLeftData = removeFeatureFromData(features, splitOnIndex, [rightData, leftData])

            # recurse child nodes of current node incrementing which features to split on
            node.right = self.fit(newRightData, rightTarget, DecisionNode(rightTarget, node), newFeatures)
            node.left = self.fit(newLeftData, leftTarget, DecisionNode(leftTarget, node), newFeatures)

            return node

    # for testing purposes
    def traverse(self, node=None):
        if node is None:
            node = self.head
        if isinstance(node, DecisionNode):
            print(node.value)

            # print(node.featureIndex)
            self.traverse(node.right)
            self.traverse(node.left)

            print('\n')

    def predict(self, data):
        return self.head.predict(data)


class DecisionNode:
    def __init__(self, value, parent):
        # pointers to parent node and child nodes
        self.parent = parent
        self.left = None
        self.right = None

        # list of target values the node was split to
        self.value = value

        # feature index that we are splitting on
        self.featureIndex = None

    def pluralityValue(self):
        return random.choice(self.value)

    def predict(self, data):
        if not data[self.featureIndex]:
            return self.left.predict(data)
        else:
            return self.right.predict(data)


class LeafNode:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, data):
        return self.prediction


class Classifier:
    def __init__(self):
        self.decisionTree = DecisionTree()

    def reset(self):
        pass

    def fit(self, data, target):
        self.decisionTree.fit(data, target)

    def predict(self, data, legal=None):
        predictedMoved = self.decisionTree.predict(data)

        if predictedMoved not in legal:
            return random.choice(legal)
        else:
            return predictedMoved


if __name__ == '__main__':
    dataS = np.loadtxt('good-moves.txt', dtype=str)
    X = [list(map(int, i[:-1])) for i in dataS]
    y = [int(i[-1]) for i in dataS]

    dt = DecisionTree()
    dt.fit(X, y)
    # print(dt.traverse())

    # proportion of training data classified correctly
    count = 0
    for i in range(len(X)):
        pred = dt.predict(X[i])
        if pred == y[i]:
            count += 1

    print(f"proportion correct: {count/126}")
