from typing import List

import dt_global
from dt_core import *
from datetime import datetime

def cv_pre_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for pre-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """  
    # value list are list of max depth
    n_folds = len(folds)
    features = dt_global.feature_names[:-1]
    num_values = len(value_list)
    training_accuracy = [0 for i in range(num_values)]
    validation_accuracy = [0 for i in range(num_values)]
    for i in range(n_folds):
        train_set = []
        validation_set = []
        for j in range(n_folds):
            if j != i:
                train_set = train_set + folds[j]
        validation_set = folds[i]
        tree = learn_dt(train_set, features)
        for i in range(num_values):
            max = value_list[i]
            train_acc = get_prediction_accuracy(tree, train_set, max)
            validation_acc = get_prediction_accuracy(tree, validation_set, max)
            training_accuracy[i] += train_acc
            validation_accuracy[i] += validation_acc
    for i in range(num_values):
        training_accuracy[i] = training_accuracy[i] / n_folds
        validation_accuracy[i] = validation_accuracy[i] / n_folds
    return training_accuracy, validation_accuracy

def cv_post_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for post-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """ 
    n_folds = len(folds)
    features = dt_global.feature_names[:-1]
    num_values = len(value_list)
    training_accuracy = [0 for i in range(num_values)]
    validation_accuracy = [0 for i in range(num_values)]
    for i in range(n_folds):
        train_set = []
        validation_set = []
        for j in range(n_folds):
            if j != i:
                train_set = train_set + folds[j]
        validation_set = folds[i]
        tree = learn_dt(train_set, features)
        for i in range(num_values):
            min = value_list[i]
            post_prune(tree, min)
            train_acc = get_prediction_accuracy(tree, train_set, min_num_examples=min)
            validation_acc = get_prediction_accuracy(tree, validation_set, min_num_examples=min)
            training_accuracy[i] += train_acc
            validation_accuracy[i] += validation_acc
    for i in range(num_values):
        training_accuracy[i] = training_accuracy[i] / n_folds
        validation_accuracy[i] = validation_accuracy[i] / n_folds
    return training_accuracy, validation_accuracy


if __name__ == "__main__":
    List_of_examples = read_data("data.csv")
    folds = preprocess(List_of_examples)

    input_feature_names = dt_global.feature_names[:-1]
    maximum_depth_value_list = []

    for i in range(0, 300, 20):
        maximum_depth_value_list.append(i)

    start = datetime.now()
    # max_training_accuracy_list, max_validation_accuracy_list = cv_post_prune(folds, maximum_depth_value_list)
    tree = learn_dt(List_of_examples, input_feature_names)
    end = datetime.now()
    print(end - start)
    # print(RenderTree(tree))
    # print(tree.height)

    # folds = preprocess(List_of_examples)


    # print(max_training_accuracy_list)
    # print(max_validation_accuracy_list)

