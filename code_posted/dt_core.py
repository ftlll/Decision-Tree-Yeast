import math
from os import major
from typing import List
from anytree import Node, RenderTree
import numpy as np

import dt_global 
from dt_provided import *
    
def labels_same(prev_labels, curr_labels):
    prev_set = set(prev_labels)
    curr_set = set(curr_labels)
    if len(prev_set) == 0:
        return True
    if len(prev_set) == 1 and len(curr_set) == 1:
        return list(prev_set)[0] == list(curr_set)[0]
    else:
        return False

def get_splits(examples: List, feature: str) -> List[float]:
    """
    Given some examples and a feature, returns a list of potential split point values for the feature.
    
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :return: a list of potential split point values 
    :rtype: List[float]
    """ 
    split_points = []
    # get the index of feature in examples
    feature_index = dt_global.feature_names.index(feature)
    examples = np.array(examples)
    # sort example by feature
    examples = examples[examples[:, feature_index].argsort()]
    prev = 0
    curr = examples[0][feature_index]
    prev_labels = []
    curr_labels = [examples[0][dt_global.label_index]]
    num_examples = len(examples)
    for i in range(num_examples):
        new_cur = examples[i][feature_index]
        if new_cur != curr:
            if not labels_same(prev_labels, curr_labels):
                split_points.append( (prev+curr) / 2)
            prev = curr
            curr = new_cur
            prev_labels = curr_labels
            curr_labels = [examples[i][dt_global.label_index]] 
        else:
            curr_labels.append(examples[i][dt_global.label_index])
    if not labels_same(prev_labels, curr_labels):
        split_points.append( (prev+curr) / 2)
    return split_points

def calculateEntropy(examples):
    label_counter = {}
    num_examples = len(examples)
    for example in examples:
        label = example[dt_global.label_index]
        if label not in label_counter.keys():
            label_counter[label] = 1
        else:
            label_counter[label] += 1
    entropy = 0.0
    for key in label_counter:
        probability = float(label_counter[key]/num_examples)  
        entropy -= float(probability * math.log(probability) / math.log(2))
    return entropy

def choose_feature_split(examples: List, features: List[str]) -> (str, float):
    """
    Given some examples and some features,
    returns a feature and a split point value with the max expected information gain.

    If there are no valid split points for the remaining features, return None and -1.

    Tie breaking rules:
    (1) With multiple split points, choose the one with the smallest value. 
    (2) With multiple features with the same info gain, choose the first feature in the list.

    :param examples: a set of examples
    :type examples: List[List[Any]]    
    :param features: a set of features
    :type features: List[str]
    :return: the best feature and the best split value
    :rtype: str, float
    """   
    best_feature = None
    best_split_point = -1
    best_GI = 0
    baseEntropy = calculateEntropy(examples)
    for feature in features:
        curr_best_gain = 0
        curr_best_split = -1
        split_points = get_splits(examples, feature)
        if len(split_points) != 0:
            for split in split_points:
                list_less, list_more = split_examples(examples, feature, split)
                less_entropy = calculateEntropy(list_less)
                more_entropy = calculateEntropy(list_more)
                less_probability = float(len(list_less)/len(examples))
                more_probability = float(len(list_more)/len(examples))
                gain_info = baseEntropy - less_probability * less_entropy - more_probability * more_entropy
                # we want the largest gain_info
                if gain_info > curr_best_gain:
                    curr_best_gain = gain_info
                    curr_best_split = split
        if curr_best_gain > best_GI:
            best_GI = curr_best_gain
            best_split_point = curr_best_split
            best_feature = feature
    return best_feature, best_split_point

def split_examples(examples: List, feature: str, split: float) -> (List, List):
    """
    Given some examples, a feature, and a split point,
    splits examples into two lists and return the two lists of examples.

    The first list of examples have their feature value <= split point.
    The second list of examples have their feature value > split point.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :param split: the split point
    :type split: float
    :return: two lists of examples split by the feature split
    :rtype: List[List[Any]], List[List[Any]]
    """ 
    list_less = []
    list_more = []
    feature_index = dt_global.feature_names.index(feature)
    for example in examples:
        if example[feature_index] <= split:
            list_less.append(example)
        elif example[feature_index] > split:
            list_more.append(example)
    return list_less, list_more

def find_majority(examples):
    label_counter = {}
    res = -1
    most_fre = -1
    for example in examples:
        label = example[dt_global.label_index]
        if label not in label_counter.keys():
            label_counter[label] = 1
        else:
            label_counter[label] += 1
    for key in label_counter:
        if label_counter[key] > most_fre:
            most_fre = label_counter[key]
            res = key
        elif label_counter[key] == most_fre:
            if key < res:
                res = key
    return res

def split_node(cur_node: Node, examples: List, features: List[str], max_depth=math.inf):
    """
    Given a tree with cur_node as the root, some examples, some features, and the max depth,
    grows a tree to classify the examples using the features by using binary splits.

    If cur_node is at max_depth, makes cur_node a leaf node with majority decision and return.

    This function is recursive.

    :param cur_node: current node
    :type cur_node: Node
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the maximum depth of the tree
    :type max_depth: int
    """ 
    feature, split = choose_feature_split(examples, features)
    if feature == None:
        decision = find_majority(examples)
        node = Node("leaf", parent=cur_node, decision=decision, size=len(examples))
    else:
        list_less, list_more = split_examples(examples, feature, split)
        if max_depth == 0:
            # reached max depth, we need to use majority
            if len(list_less) >= len(list_more):
                decision = find_majority(list_less)
            else:
                decision = find_majority(list_more)
            node = Node("leaf", parent=cur_node, decision=decision, size=len(examples))
        else:
            max_depth -= 1
            decision = find_majority(examples)
            node = Node("idk", parent = cur_node ,feature = feature, split = split, 
                size=len(examples), decision = decision)
            # left for less than
            split_node(node, list_less, features, max_depth)
            # right for more than
            split_node(node, list_more, features, max_depth)

def learn_dt(examples: List, features: List[str], max_depth=math.inf) -> Node:
    """
    Given some examples, some features, and the max depth,
    creates the root of a decision tree, and
    calls split_node to grow the tree to classify the examples using the features, and
    returns the root node.

    This function is a wrapper for split_node.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the max depth of the tree
    :type max_depth: int, default math.inf
    :return: the root of the tree
    :rtype: Node
    """ 
    feature, split = choose_feature_split(examples, features)
    root_node = Node("root", feature=feature, split=split, 
        size=len(examples), decision=find_majority(examples))
    list_less, list_more = split_examples(examples, feature, split)
    split_node(root_node, list_less, features, max_depth - 1)
    split_node(root_node, list_more, features, max_depth - 1)
    return root_node

def predict(cur_node: Node, example, max_depth=math.inf, \
    min_num_examples=0) -> int:
    """
    Given a tree with cur_node as its root, an example, and optionally a max depth,
    returns a prediction for the example based on the tree.

    If max_depth is provided and we haven't reached a leaf node at the max depth, 
    return the majority decision at this node.

    If min_num_examples is provided and the number of examples at the node is less than min_num_examples, 
    return the majority decision at this node.
    
    This function is recursive.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param cur_node: cur_node of a decision tree
    :type cur_node: Node
    :param example: one example
    :type example: List[Any]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the decision for the given example
    :rtype: int
    """ 
    decision = cur_node.decision
    # if it is a leaf, just stop it
    if len(cur_node.children) == 0:
        return decision
    # if it is not a leaf, check the optional parameter
    if max_depth == 0:
        return decision
    size = cur_node.size
    if min_num_examples != 0 and size < min_num_examples:
        return decision

    feature_index = dt_global.feature_names.index(cur_node.feature)

    if example[feature_index] <= cur_node.split:
        index = 0
    else:
        index = 1
    return predict(cur_node.children[index], example, max_depth - 1, min_num_examples)

def get_prediction_accuracy(cur_node: Node, examples: List, max_depth=math.inf, \
    min_num_examples=0) -> float:
    """
    Given a tree with cur_node as the root, some examples, 
    and optionally the max depth or the min_num_examples, 
    returns the accuracy by predicting the examples using the tree.

    The tree may be pruned by max_depth or min_num_examples.

    :param cur_node: cur_node of the decision tree
    :type cur_node: Node
    :param examples: the set of examples. 
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the prediction accuracy for the examples based on the cur_node
    :rtype: float
    """ 
    num = len(examples)
    num_correct = 0
    for example in examples:
        prediction = predict(cur_node, example, max_depth, min_num_examples)
        label = example[dt_global.label_index]
        if prediction == label:
            num_correct += 1
    return float(num_correct / num)

def post_prune(cur_node: Node, min_num_examples: float):
    """
    Given a tree with cur_node as the root, and the minimum number of examples,
    post prunes the tree using the minimum number of examples criterion.

    This function is recursive.

    Let leaf parents denote all the nodes that only have leaf nodes as its descendants. 
    Go through all the leaf parents.
    If the number of examples at a leaf parent is smaller than the pre-defined value,
    convert the leaf parent into a leaf node.
    Repeat until the number of examples at every leaf parent is greater than
    or equal to the pre-defined value of the minimum number of examples.

    :param cur_node: the current node
    :type cur_node: Node
    :param min_num_examples: the minimum number of examples
    :type min_num_examples: float
    """
    # if we go to this step, it means for cur_node, size > min_num_examples
    if len(cur_node.children) == 0:
        return cur_node
    else:
        size = cur_node.size
        children = cur_node.children
        decision = cur_node.decision
        if size < min_num_examples:
            # we should not split
            new_size = children[0].size + children[1].size
            cur_node.name = "leaf"
            cur_node.size = new_size
            cur_node.decision = decision
            cur_node.children = []
            del cur_node.feature
            del cur_node.split
        else:
            post_prune(children[0], min_num_examples)
            post_prune(children[1], min_num_examples)
