# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import numpy as np
import random


def featureIndexToSplitOn(features, data, target):
    split_index = [gini(feature_index, data, target) for feature_index in range(len(features))]
    min_index = split_index.index(min(split_index))

    return min_index


def gini(feature_index, data, target):
    one, zero = [], []
    target_counter = [[0] * 4, [0] * 4]

    for i in list(zip(data, target)):
        (one if int(i[0][feature_index]) else zero).append(i)
        target_counter[int(i[0][feature_index])][int(i[1])] += 1

    si = [len(one), len(zero)]
    s = sum(si)
    gini_impurity = 0

    for i in range(2):
        gi = sum([j ** 2 for j in target_counter[i]])
        gi = 1 - (gi / (s ** 2))
        gini_impurity += (si[i] / s) * gi

    return gini_impurity


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
            self.head = DecisionNode(data, None)
            node = self.head
            features = list(range(len(data[0])))

        # if there are no example in this branch then make it a leaf with its parents plurality value for class
        if len(data) == 0:
            return LeafNode(node.parent.pluralityValue())

        # keep this brain hurt but keep
        # elif len(set(data)) == 1:
        #     return Leaf(node.pluralityValue())

        # if there are no more features to split on
        elif len(features) == 0:
            return LeafNode(node.pluralityValue())

        # if all the example have the same classification then make leaf with that classification
        elif len(set(target)) == 1:
            return LeafNode(target[0])

        else:
            split_on = featureIndexToSplitOn(features, data, target)
            rightData, rightTarget, leftData, leftTarget = [], [], [], []

            node.featureIndex = features[split_on]

            # go through data and split on features --> binary split
            for i in list(zip(data, target)):
                if int(i[0][split_on]):
                    rightData.append(i[0])
                    rightTarget.append(i[1])
                else:
                    leftData.append(i[0])
                    leftTarget.append(i[1])

            newFeatures, newRightData, newLeftData = removeFeatureFromData(features, split_on, [rightData, leftData])

            # recurse child nodes of current node incrementing which features to split on
            # can change right_target to just right but do at end
            node.right = self.fit(newRightData, rightTarget, DecisionNode(rightTarget, node), newFeatures)
            node.left = self.fit(newLeftData, leftTarget, DecisionNode(leftTarget, node), newFeatures)

            return node

    # for testing purposes
    def traverse(self, node):
        if isinstance(node, LeafNode):
            print(node.prediction)
        else:
            # print(node.featureIndex)
            self.traverse(node.right)
            self.traverse(node.left)

    def predict(self):
        pass


class DecisionNode:
    def __init__(self, value, parent):
        # pointers to parent node and child nodes
        self.parent = parent
        self.left = None
        self.right = None

        # value is all the target values the node took
        self.value = value

        # feature index that we are splitting on
        self.featureIndex = None

    def pluralityValue(self):
        return random.choice(self.value)

    def predict(self):
        pass


class LeafNode:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self):
        return self.prediction


class Classifier:
    def __init__(self):
        self.decisionTree = DecisionTree()

    def reset(self):
        pass

    def fit(self, data, target):
        self.decisionTree.fit(data, target)

    def predict(self, data, legal=None):
        return 1


if __name__ == '__main__':
    dataS = np.loadtxt('good-moves.txt', dtype=str)
    X = [i[:-1] for i in dataS]
    y = [i[-1] for i in dataS]

    dt = DecisionTree()
    dt.fit(X, y)

    dt.traverse(dt.head)
