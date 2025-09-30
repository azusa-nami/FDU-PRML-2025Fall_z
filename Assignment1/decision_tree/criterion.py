"""
criterion
"""

import math
import numpy as np


def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels



def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    parent_entropy=0
    for label,count in all_labels.items():
        parent_entropy-=count/y.shape[0]*math.log2(count/y.shape[0])

    left_entropy=0
    for label,count in left_labels.items():
        left_entropy-=count/l_y.shape[0]*math.log2(count/l_y.shape[0])

    right_entropy=0
    for label,count in right_labels.items():
        right_entropy-=count/r_y.shape[0]*math.log2(count/r_y.shape[0])

    info_gain=parent_entropy-(l_y.shape[0]/y.shape[0])*left_entropy-(r_y.shape[0]/y.shape[0])*right_entropy
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ratio1 = l_y.shape[0] / y.shape[0]
    ratio2 = r_y.shape[0] / y.shape[0]
    if ratio1 == 0:
        ratio1=1e-5
    if ratio2 == 0:
        ratio2=1e-5
    try:
        split_info = -ratio1 * math.log2(ratio1) - ratio2 * math.log2(ratio2)
        gain_ratio = info_gain / split_info
        return gain_ratio
    except Exception as e:
        print(ratio1)
        print(ratio2)
        print(e)
        exit(-1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    all_labels=np.array(list(all_labels.values()))
    left_labels=np.array(list(left_labels.values()))
    right_labels=np.array(list(right_labels.values()))

    all_label_prob=all_labels/all_labels.sum()
    left_label_prob=left_labels/left_labels.sum()
    right_label_prob=right_labels/right_labels.sum()

    before=1-np.sum(all_label_prob*all_label_prob)
    left=1-np.sum(left_label_prob*left_label_prob)
    right=1-np.sum(right_label_prob*right_label_prob)

    after=left*l_y.shape[0]/y.shape[0]+right*r_y.shape[0]/y.shape[0]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    if len(left_labels)==0 or len(right_labels)==0:
        return -1

    all_labels=np.array(list(all_labels.values()))
    left_labels=np.array(list(left_labels.values()))
    right_labels=np.array(list(right_labels.values()))

    before=np.max(all_labels)/np.sum(all_labels)
    left=np.max(left_labels)/np.sum(left_labels)
    right=np.max(right_labels)/np.sum(right_labels)

    after=left*l_y.shape[0]/y.shape[0]+right*r_y.shape[0]/y.shape[0]

    result = after - before
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return result
