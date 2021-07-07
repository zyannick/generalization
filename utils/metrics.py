import torch

import random

import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def retrieve_gt_pd(per_frame_logits, labels):
    y_predicted = per_frame_logits
    y_predicted = y_predicted.cuda()
    y_predicted = y_predicted.cpu().detach().numpy()
    y_predicted = np.argmax(y_predicted, axis=1)

    y_ground_truth = labels
    y_ground_truth = y_ground_truth.cuda()
    y_ground_truth = y_ground_truth.cpu().detach().numpy()

    return y_ground_truth, y_predicted


def epoch_accuracy_fscore(y_ground_truth, y_predicted):
    from sklearn.metrics import multilabel_confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    return accuracy_score(y_ground_truth, y_predicted), \
           f1_score(y_ground_truth, y_predicted, average='weighted', labels=np.unique(y_predicted))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cross_batch_accuracy(per_frame_logits, labels):
    y_predicted = torch.max(per_frame_logits, dim=2)[0]
    y_predicted = y_predicted.cuda()
    y_predicted = y_predicted.cpu().detach().numpy()
    y_predicted = np.argmax(y_predicted, axis=1)

    y_ground_truth = torch.max(labels, dim=2)[0]
    y_ground_truth = y_ground_truth.cuda()
    y_ground_truth = y_ground_truth.cpu().detach().numpy()
    y_ground_truth = np.argmax(y_ground_truth, axis=1)

    # print(y_predicted)
    # print(y_ground_truth)
    # print('\n')

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    # print(multilabel_confusion_matrix(y_ground_truth, y_predicted))
    # print('Cross batch accuracy %4.4f'%(accuracy_score(y_ground_truth, y_predicted)))
    return accuracy_score(y_ground_truth, y_predicted), f1_score(y_ground_truth, y_predicted, average='weighted')


def per_video_accuracy(per_frame_logits, labels):
    y_predicted = per_frame_logits.cuda()
    y_predicted = y_predicted.cpu().detach().numpy()

    y_ground_truth = labels.cuda()
    y_ground_truth = y_ground_truth.cpu().detach().numpy()

    nb_batches = y_predicted.shape[0]

    i = random.randint(0, nb_batches - 1)

    r_shape = y_predicted.shape

    y_predicted = y_predicted[i]
    y_ground_truth = y_ground_truth[i]

    y_predicted = np.reshape(y_predicted, (r_shape[1], r_shape[2]))
    y_ground_truth = np.reshape(y_ground_truth, (r_shape[1], r_shape[2]))

    y_predicted = np.argmax(y_predicted, axis=0)
    y_ground_truth = np.argmax(y_ground_truth, axis=0)

    from sklearn.metrics import accuracy_score
    print('Per video batch accuracy %4.4f' % (accuracy_score(y_ground_truth, y_predicted)))