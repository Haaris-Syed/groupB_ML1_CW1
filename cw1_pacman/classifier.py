# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

class Classifier:
    def __init__(self):
        self.decisionTree = DecisionTree()

    def reset(self):
        pass
    
    def fit(self, data, target):
        self.decisionTree.fit(data, target)

    def predict(self, data, legal=None):
        return 1
        
class DecisionTree:
    def __init__(self) -> None:
        self.head = None

    def fit(self, data, target):
        pass

    def predict(self):
        pass

class Node:
    def __init__(self) -> None:
        # pointers to child nodes
        self.children = []
        # feature index that we are splitting on
        self.featureIndex = None

    def predict(self):
        pass

class Leaf:
    def __init__(self) -> None:
        self.prediction = None

    def predict(self):
        return self.prediction