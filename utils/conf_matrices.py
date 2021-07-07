import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns
import os



list_classes = ['walk', 'pick', 'lie_sit', 'standup', 'crawl',
                'lie_fall', 'stfall', 'fall']

def label_ordering(label):
    if label in [0, 1, 2, 3, 4]:
        return 0
    else:
        return 1

def correct_labels(labels):
    for i in range(labels.shape[0]):
        labels[i] = label_ordering(labels[i])
    return labels

def plot_results_inference(two_classes):

    root_dir = 'infer_results'

    # method = 'rgb_2_edgetir_normal_1input_'

    y_ground_truth = pd.read_csv(os.path.join(os.getcwd(), root_dir, 'ground_truth_tir_sobel_rgb_flow_0.csv'), sep=';', header=None)
    y_ground_truth = y_ground_truth.values

    y_predicted = pd.read_csv(os.path.join(os.getcwd(), root_dir, 'predict_tir_sobel_rgb_flow_0.csv'), sep=';', header=None)
    y_predicted = y_predicted.values

    if two_classes:
        y_ground_truth = correct_labels(y_ground_truth)
        y_predicted = correct_labels(y_predicted)

    print('Accuracy %4.4f' % (accuracy_score(y_ground_truth, y_predicted)))
    print('F1score %4.4f' % (f1_score(y_ground_truth, y_predicted, average='weighted')))

    if two_classes:
        return

    cm = confusion_matrix(y_target=y_ground_truth,
                          y_predicted=y_predicted,
                          binary=False)

    font = {'family': 'serif',
            'size': 25}

    matplotlib.rc('font', **font)
    plt.figure(figsize=(16, 16))
    fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=list_classes, figsize=(16, 16))

    where_to_save = os.path.join(os.getcwd(), 'plotted_images')
    if not os.path.exists(where_to_save):
        os.makedirs(where_to_save)
    plt.savefig(os.path.join(where_to_save, 'tir_sobel_rgb_flow.png'))
    # plt.show()


if __name__ == '__main__':
    two_classes = True
    plot_results_inference(two_classes)



