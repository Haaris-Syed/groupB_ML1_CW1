# classifier.py
# Lin Li/26-dec-2021
import numpy as np
import math
import random
import copy


def getFeatureIndexToSplitOn(features, data, target, featureSize):
    numberOfFeatures = random.randint(5, 15)
    splitIndex = [gini([random.choice(list(range(len(features)))) for _ in range(featureSize)], data, target)
                  for _ in range(numberOfFeatures)]
    minIndexSubset = min(splitIndex, key=lambda x: x[1])

    return minIndexSubset[0]


def infoGain(data, featureIndex, target):
    one, zero = [], []
    targetCounter = [[0] * 4, [0] * 4]

    for i in list(zip(data, target)):
        (one if int(i[0][featureIndex]) else zero).append(i)
        targetCounter[int(i[0][featureIndex])][int(i[1])] += 1

    entropy = lambda x: -sum([pi * math.log2(pi) for pi in x])

    hs = entropy([len(one), len(zero)])
    s = len(one) + len(zero)
    targetCounterZip = list(zip(targetCounter))
    gain = 0

    for i in targetCounterZip:
        gain += (sum(i) / s) * entropy(i)

    gain = hs - gain

    return gain


def gini(featureIndexes, data, target):
    examplePerms = dict()
    targetCounter = dict()

    for i in list(zip(data, target)):
        key = ''.join([str(i[0][index]) for index in featureIndexes])
        examplePerms[key] = examplePerms.get(key, 0) + 1
        if key in targetCounter:
            targetCounter[key][int(i[1])] = targetCounter.get(key, dict()).get(int(i[1]), 0) + 1
        else:
            targetCounter[key] = dict()

    s = sum(examplePerms.values())
    giniImpurity = 0

    for key in targetCounter:
        gi = sum([j ** 2 for j in targetCounter[key]])
        gi = 1 - (gi / (s ** 2))
        giniImpurity += (examplePerms[key] / s) * gi

    return [featureIndexes, giniImpurity]


def removeFeatureFromData(features, featureIndex, datas):
    for key in datas:
        print(key)
        print(featureIndex)
        for i in range(len(datas[key])):
            for feature in featureIndex:
                def datas[key][i][0][feature]
            datas[key][i][0] = ''.join([list(datas[key][i]).pop(feature) for feature in featureIndex])
            print(datas[key][0])

    [features.pop(feature) for feature in featureIndex]

    return features, datas


class DecisionTree:
    """A single decision tree."""

    def __init__(self):
        self.head = None

    def fit(self, data, target, node=None, features=None):

        # check if node is the root node
        if node is None:
            self.head = DecisionNode(target, None)
            node = self.head
            features = list(range(len(data[0])))

        # if all the example have the same classification then make leaf with that classification
        if len(set(target)) == 1:
            return LeafNode(target[0])

        # if there are no example in this branch then make it a leaf with its parents plurality value for class
        elif len(data) == 0:
            return LeafNode(node.parent.pluralityValue())

        # if there are no more features to split on
        elif len(features) == 0:
            return LeafNode(node.pluralityValue())

        else:
            splitOnIndex = getFeatureIndexToSplitOn(features, data, target, 2)
            # rightData, rightTarget, leftData, leftTarget = [], [], [], []
            node.featureIndex = [features[index] for index in splitOnIndex]
            datas = dict()

            for i in list(zip(data, target)):
                key = ''.join([str(i[0][index]) for index in splitOnIndex])

                if key in datas:
                    datas[key].append([i])
                else:
                    datas[key] = list()

            print(datas)

            # # go through data and split on features
            # for i in list(zip(data, target)):
            #     if int(i[0][splitOnIndex]):
            #         rightData.append(i[0])
            #         rightTarget.append(i[1])
            #     else:
            #         leftData.append(i[0])
            #         leftTarget.append(i[1])

            newFeatures, newData = removeFeatureFromData(features, splitOnIndex, datas)

            # recurse child nodes of current node incrementing which features to split on
            # node.right = self.fit(newRightData, rightTarget, DecisionNode(rightTarget, node), newFeatures)
            # node.left = self.fit(newLeftData, leftTarget, DecisionNode(leftTarget, node), newFeatures)
            #
            # return node

    def draw_node(self, node=None, level=0, right=''):
        if node is None:
            node = self.head
        if isinstance(node, LeafNode):
            # Draw the label of the leaf node using 'l' and the appropriate indentation
            print('|   ' * level + '|-- L' + right)
        else:
            # Draw the condition of the non-leaf node using 'o' and the appropriate indentation
            print('|   ' * level + '|-- D' + right)
            # Recursively draw the child nodes
            self.draw_node(node.left, level + 1, 'l')
            self.draw_node(node.right, level + 1,'r')

    def predict(self, data):
        if self.head is not None:
            return self.head.predict(data)
        return 0

    def accuracy(self, X, y):
        return sum(int(self.predict(X[i]) == y[i]) for i in range(len(X))) / len(X)

    def prune(self, xValidation, yValidation, node=None, right=None):
        if node is None:
            node = self.head

        # check if traversed to bottom of tree returning the accuracy of the tree at that point
        if isinstance(node, LeafNode):
            return self.accuracy(xValidation, yValidation)
        else:
            # using the accuracy calculated at the next recurs step used to compare performance of tree
            priorAccuracyR = self.prune(xValidation, yValidation, node.right, 1)
            priorAccuracyL = self.prune(xValidation, yValidation, node.left, 0)

            # perform pruning checking on condition and altering tree
            if node != self.head:
                tempNode = copy.copy(node)
                newLeaf = LeafNode(node.pluralityValue)
                node.parent.set(right, newLeaf)
                newAccuracy = self.accuracy(xValidation, yValidation)

                if newAccuracy > max(priorAccuracyR, priorAccuracyL):
                    del tempNode
                    print(newAccuracy)
                    return newAccuracy
                else:
                    node.parent.set(right, tempNode)
                    return priorAccuracyR if right else priorAccuracyL


class DecisionNode:
    """A decision node (non leaf node) in a decision tree."""

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

    def set(self, right, node):
        if right:
            self.right = node
        else:
            self.left = node

    def predict(self, data):
        if int(data[self.featureIndex]):
            return self.right.predict(data)
        else:
            return self.left.predict(data)


class LeafNode:
    """A leaf node in a decision tree.
    
    Represents the prediction returned by the decision tree"""

    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, data):
        return self.prediction


class Classifier:
    def __init__(self):
        self.decisionTrees = [DecisionTree() for _ in range(5)]

    def reset(self):
        pass

    def fit(self, data, target):
        # fit all decision trees
        # we apply bagging
        for i in range(len(self.decisionTrees)):
            data_i, target_i = self.createTrainingSet(data, target)
            self.decisionTrees[i].fit(data_i, target_i)

    def createTrainingSet(self, data, target):
        data_i = []
        target_i = []

        for _ in range(len(data)):
            data_i.append(random.choice(data))
            target_i.append(random.choice(target))

        return data_i, target_i

    def predict(self, data, legal=None):
        predictions = [tree.predict(data) for tree in self.decisionTrees]
        prediction_counts = [predictions.count(p) for p in range(4)]

        # take predicted move as the move with the most votes by the decision trees -> hard voting
        predictedMove = prediction_counts.index(max(prediction_counts))

        if predictedMove not in legal:
            return random.choice(legal)
        else:
            return predictedMove


if __name__ == '__main__':
    dataS = np.loadtxt('good-moves.txt', dtype=str)
    training = dataS[:-30]
    testing = [i for i in dataS if i not in training]

    X = [[int(c) for c in i[:-1]] for i in training]
    y = [int(i[-1]) for i in training]
    X_test = [[int(c) for c in i[:-1]] for i in testing]
    y_test = [int(i[-1]) for i in testing]

    dt = DecisionTree()
    dt.fit(X, y)
    # print(dt.accuracy(X_test, y_test))
    # dt.prune(X_test, y_test)
    # print(dt.accuracy(X_test, y_test))
