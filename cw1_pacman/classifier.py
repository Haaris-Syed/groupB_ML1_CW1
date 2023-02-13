# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import numpy as np
import random


class DecisionTree:
    def __init__(self):
        self.head = None

    def fit(self, data, target, node=None, split_on=None):

        # check if node is the root node
        if node is None:
            self.head = DecisionNode(data, None)
            split_on = 0
            node = self.head

        # if there are no example in this branch then make it a leaf with its parents plurality value for class
        if len(data) == 0:
            return LeafNode(node.parent.pluralityValue())

        # keep this brain hurt but keep
        # elif len(set(data)) == 1:
        #     return Leaf(node.pluralityValue())

        # if there are no more features to split on
        elif split_on == 24:
            return LeafNode(node.pluralityValue())

        # if all the example have the same classification then make leaf with that classification
        elif len(set(target)) == 1:
            return LeafNode(target[0])

        else:
            right = []
            left = []

            # go through data and split on features --> binary split
            for i in list(zip(data, target)):
                (right if int(i[0][split_on]) else left).append(i)

            rightData, rightTarget, leftData, leftTarget = [], [], [], []

            if len(right) != 0:
                rightData, rightTarget = zip(*right)

            if len(left) != 0:
                leftData, leftTarget = zip(*left)

            # recurse child nodes of current node incrementing which features to split on
            node.right = self.fit(rightData, rightTarget, Node(rightTarget, node), split_on+1)
            node.left = self.fit(leftData, leftTarget, Node(leftTarget, node), split_on+1)

            node.featureIndex = split_on

            return node

    # for testing purposes
    def traverse(self, node):
        if isinstance(node, Leaf):
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