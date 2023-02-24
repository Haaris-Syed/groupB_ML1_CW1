# classifier.py
# Lin Li/26-dec-2021
import numpy as np
import math
import random
import copy


def getFeatureIndexToSplitOn(features, data, target):
    # randomly choose between using Gini impurity or information gain to find the feature to split on
    if random.randint(0, 1) == 0:
        featureGiniImpurities = [gini(featureIndex, data, target) for featureIndex in range(len(features))]
        index = featureGiniImpurities.index(min(featureGiniImpurities))
    else:
        featureInfoGainValues = [infoGain(featureIndex, data, target) for featureIndex in range(len(features))]
        index = featureInfoGainValues.index(max(featureInfoGainValues))

    return index


def infoGain(featureIndex, data, target):
    one, zero = [], []
    targetCounter = [[0] * 4, [0] * 4]

    for i in list(zip(data, target)):
        (one if int(i[0][featureIndex]) else zero).append(i)
        targetCounter[int(i[0][featureIndex])][int(i[1])] += 1

    entropy = lambda x: -sum([pi * math.log2(pi) for pi in x if pi > 0])

    hs = entropy([len(one), len(zero)])
    s = len(one) + len(zero)
    targetCounterZip = list(zip(targetCounter))
    gain = 0

    for i in targetCounterZip:
        gain += (sum(i[0]) / s) * entropy(i[0])

    gain = hs - gain

    return gain


def gini(featureIndex, data, target):
    # initialise one and zero example lists
    one, zero = [], []
    targetCounter = [[0] * 4, [0] * 4]

    # append examples to correct list and count labels in each example
    for i in list(zip(data, target)):
        (one if int(i[0][featureIndex]) else zero).append(i)
        targetCounter[int(i[0][featureIndex])][int(i[1])] += 1

    si = [len(one), len(zero)]
    s = sum(si)
    giniImpurity = 0

    # calculate the gini value
    for i in range(2):
        gi = sum([j ** 2 for j in targetCounter[i]])
        gi = 1 - (gi / (s ** 2))
        giniImpurity += (si[i] / s) * gi

    return giniImpurity


def removeFeatureFromData(features, feature_index, data):
    # after splitting on a feature, we remove it from the feature vector
    for i in range(2):
        for j in range(len(data[i])):
            data[i][j] = data[i][j][:feature_index] + data[i][j][feature_index + 1:]

    features.pop(feature_index)

    return features, data[0], data[1]


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
            # take root n feature => root(len(features)) => ~5
            featureSubset = random.sample(features, min(len(features), 5))
            splitOnIndex = getFeatureIndexToSplitOn(featureSubset, data, target)
            rightData, rightTarget, leftData, leftTarget = [], [], [], []
            node.featureIndex = featureSubset[splitOnIndex]

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

    def predict(self, data):
        if self.head is not None:
            return self.head.predict(data)
        return 0

    def accuracy(self, X, y):
        # calculates the accuracy of the decision tree
        # accuracy = correct predictions / total predictions
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

                # if the tree has a better accuracy (i.e. performs better) when the node is pruned,
                # then we prune the node, else we keep it.
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
        self.NUM_DECISION_TREES = 64
        self.decisionTrees = [DecisionTree() for _ in range(self.NUM_DECISION_TREES)]

    def reset(self):
        self.decisionTrees = [DecisionTree() for _ in range(self.NUM_DECISION_TREES)]

    def fit(self, data, target):
        # fit all decision trees
        # we apply bagging
        for i in range(self.NUM_DECISION_TREES):
            data_i, target_i = self.createTrainingSet(data, target)
            self.decisionTrees[i].fit(data_i, target_i)

    def createTrainingSet(self, data, target):
        """Create a bagged training set"""

        data_i = []
        target_i = []

        for _ in range(len(data)):
            data_i.append(random.choice(data))
            target_i.append(random.choice(target))

        return data_i, target_i

    def predict(self, data, legal=None):
        predictions = [tree.predict(data) for tree in self.decisionTrees]

        # number of decision trees that have predicted each of the four classes
        prediction_counts = [predictions.count(p) for p in range(4)]

        # take predicted move as the move with the most votes by the decision trees
        predictedMove = prediction_counts.index(max(prediction_counts))

        if predictedMove not in legal:
            return random.choice(legal)
        else:
            return predictedMove
