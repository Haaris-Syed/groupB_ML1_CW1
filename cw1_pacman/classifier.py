# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import numpy as np
import random


class DecisionTree:
    def __init__(self):
        self.head = None

    def fit(self, data, target, node=None):

        # check if node is the root node
        if node is None:
            self.head = DecisionNode(data, None)
            node = self.head

        # if there are no example in this branch then make it a leaf with its parents plurality value for class
        if len(data) == 0:
            return LeafNode(node.parent.pluralityValue())

        # keep this brain hurt but keep
        # elif len(set(data)) == 1:
        #     return Leaf(node.pluralityValue())

        # if there are no more features to split on
        elif len(set(data)) == 1:
            return LeafNode(node.pluralityValue())

        # if all the example have the same classification then make leaf with that classification
        elif len(set(target)) == 1:
            return LeafNode(target[0])

        else:
            # right = []
            # left = []

            splitOnIndex = self.featureIndexToSplitOn(data, target)

            rightData, rightTarget, leftData, leftTarget = [], [], [], []
            
            # go through data and split on features --> binary split
            for i in list(zip(data, target)):
                #(right if int(i[0][splitOnIndex]) else left).append(i)
                if int(i[0][splitOnIndex]):
                    rightData.append(i[0])
                    rightTarget.append(i[1])
                else:
                    leftData.append(i[0])
                    leftTarget.append(i[1])

            # if len(right) != 0:
            #     rightData, rightTarget = zip(*right)

            # if len(left) != 0:
            #     leftData, leftTarget = zip(*left)

            # recurse child nodes of current node incrementing which features to split on
            node.right = self.fit(rightData, rightTarget, DecisionNode(rightTarget, node))
            node.left = self.fit(leftData, leftTarget, DecisionNode(leftTarget, node))

            node.featureIndex = splitOnIndex

            return node

    # for testing purposes
    def traverse(self, node):
        if isinstance(node, LeafNode):
            print(node.prediction)
        else:
            # print(node.featureIndex)
            self.traverse(node.right)
            self.traverse(node.left)

    def predict(self, data):
        self.head.predict(data)


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

    def predict(self, data):
        if data[self.featureIndex] == 0:
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
        # use legal moves??
        return self.decisionTree.predict(data)


if __name__ == '__main__':
    dataS = np.loadtxt('good-moves.txt', dtype=str)
    X = [i[:-1] for i in dataS]
    y = [i[-1] for i in dataS]

    dt = DecisionTree()
    dt.fit(X, y)

    dt.traverse(dt.head)