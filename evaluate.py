import numpy as np
from scipy import spatial
import os
import argparse
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score


def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = float(np.sum((prediction == 0) & (groundtruth == 0)))

    return FP, FN, TP, TN


def Accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0


def Precision_score(inp, target):
    input_flatten = inp.flatten()
    target_flatten = target.flatten()
    return precision_score(target_flatten, input_flatten)


def Recall_score(inp, target):
    input_flatten = inp.flatten()
    target_flatten = target.flatten()
    return recall_score(target_flatten, input_flatten)


def F1_score(inp, target):
    input_flatten = inp.flatten()
    target_flatten = target.flatten()
    return f1_score(target_flatten, input_flatten)


def Jaccard_score(inp, target):
    input_flatten = inp.flatten()
    target_flatten = target.flatten()
    return jaccard_score(target_flatten, input_flatten)


class evaluate_VOS():

    def __init__(self, epoch_root, gt_root, best_log, result_log):
        self.epoch_root = epoch_root
        self.gt_root = gt_root
        self.best_log = best_log
        self.result_log = result_log

        self.gt_video_names = os.listdir(self.gt_root)

        self.epoch_pathes = []
        epoch_names = os.listdir(self.epoch_root)
        for epoch_name in epoch_names:
            path = os.path.join(epoch_root, epoch_name)
            self.epoch_pathes.append(path)

    def evaluate_all_epoches(self):
        best_dice = 0
        best_epoch = ''
        for epoch_path in self.epoch_pathes:
            print('Evaluating {}'.format(epoch_path))
            dice_epoch, accuracy_epoch, precision_epoch, recall_epoch, f_measure_epoch, jaccard_epoch = self.evaluate_one_epoch(epoch_path)
            best_epoch, best_dice = self.write_result_log(epoch_path, self.result_log, dice_epoch, accuracy_epoch,
                                                        precision_epoch, recall_epoch, f_measure_epoch, jaccard_epoch,
                                                        best_dice, best_epoch, self.best_log)
            print('Best epoch path:{0}\nbest_dice:{1}'.format(best_epoch, best_dice))

    def evaluate_one_epoch(self, epoch_path):

        dice_lst, accuracy_lst, precision_lst, recall_lst, f_measure_lst, jaccard_lst = [], [], [], [], [], []

        for gt_video_name in self.gt_video_names:

            epoch_video_path = os.path.join(epoch_path, gt_video_name)
            gt_video_path = os.path.join(self.gt_root, gt_video_name)
            dice_video, accuracy_video, precision_video, recall_video, f_measure_video, jaccard_video = self.evaluate_one_video(epoch_video_path, gt_video_path)

            dice_lst.append(dice_video)
            accuracy_lst.append(accuracy_video)
            precision_lst.append(precision_video)
            recall_lst.append(recall_video)
            f_measure_lst.append(f_measure_video)
            jaccard_lst.append(jaccard_video)

        dice_epoch = sum(dice_lst) / len(dice_lst)
        accuracy_epoch = sum(accuracy_lst) / len(accuracy_lst)
        precision_epoch = sum(precision_lst) / len(precision_lst)
        recall_epoch = sum(recall_lst) / len(recall_lst)
        f_measure_epoch = sum(f_measure_lst) / len(f_measure_lst)
        jaccard_epoch = sum(jaccard_lst) / len(jaccard_lst)

        return dice_epoch, accuracy_epoch, precision_epoch, recall_epoch, f_measure_epoch, jaccard_epoch

    def evaluate_one_img(self, pred, gt):

        dice = dice_coeff(pred, gt)
        accuracy = Accuracy_score(pred, gt)
        precision = Precision_score(pred, gt)
        recall = Recall_score(pred, gt)
        f_measure = F1_score(pred, gt)
        jaccard = Jaccard_score(pred, gt)

        return dice, accuracy, precision, recall, f_measure, jaccard

    def evaluate_one_video(self, epoch_video_path, gt_video_path):

        dice_lst, accuracy_lst, precision_lst, recall_lst, f_measure_lst, jaccard_lst = [], [], [], [], [], []

        pred_names = os.listdir(epoch_video_path)
        gt_names = os.listdir(gt_video_path)

        pred_pathes = []
        gt_pathes = []

        assert len(pred_names) == len(gt_names)
        for idx in range(len(pred_names)):
            pred_name, gt_name = pred_names[idx], gt_names[idx]
            path = os.path.join(epoch_video_path, pred_name)
            pred_pathes.append(path)
            path = os.path.join(gt_video_path, gt_name)
            gt_pathes.append(path)

        assert len(pred_pathes) == len(gt_pathes)
        for idx in range(len(pred_pathes)):
            pred_path, gt_path = pred_pathes[idx], gt_pathes[idx]
            pred = np.array(Image.open(pred_path))
            gt = np.array(Image.open(gt_path))

            dice, accuracy, precision, recall, f_measure, jaccard = self.evaluate_one_img(pred, gt)

            dice_lst.append(dice)
            accuracy_lst.append(accuracy)
            precision_lst.append(precision)
            recall_lst.append(recall)
            f_measure_lst.append(f_measure)
            jaccard_lst.append(jaccard)

            dice_video = sum(dice_lst) / len(dice_lst)
            accuracy_video = sum(accuracy_lst) / len(accuracy_lst)
            precision_video = sum(precision_lst) / len(precision_lst)
            recall_video = sum(recall_lst) / len(recall_lst)
            f_measure_video = sum(f_measure_lst) / len(f_measure_lst)
            jaccard_video = sum(jaccard_lst) / len(jaccard_lst)

        return dice_video, accuracy_video, precision_video, recall_video, f_measure_video, jaccard_video

    def write_result_log(self, epoch_path, result_log, dice_epoch, accuracy_epoch, precision_epoch, recall_epoch,
                         f_measure_epoch, jaccard_epoch, best_dice, best_epoch, best_log):
        print('Model: {0}'.format(epoch_path))
        print('Dice: {:.6f}\nAccuracy: {:.6f}\nPrecision: {:.6f}\nRecall: {:.6f}\nf_measure: {:.6f}\nJaccard: {:.6f}\n '.format(
                            dice_epoch, accuracy_epoch, precision_epoch, recall_epoch, f_measure_epoch, jaccard_epoch))
        open(result_log, 'a').write('Model: {0}\n'.format(epoch_path))
        open(result_log, 'a').write('Dice: {:.6f}\nAccuracy: {:.6f}\nPrecision: {:.6f}\nRecall: {:.6f}\nf_measure: {:.6f}\nJaccard: {:.6f}\n '.format(
                            dice_epoch, accuracy_epoch, precision_epoch, recall_epoch, f_measure_epoch, jaccard_epoch))

        if best_dice < dice_epoch:
            best_dice = dice_epoch
            best_epoch = epoch_path
            open(best_log, 'a').write('Best epoch path:{0}\n'.format('Best epoch path:{0}\nbest_dice:{1}'.format(best_epoch, best_dice)))
            print('Best epoch path:{0}\nbest_dice:{1}'.format(best_epoch, best_dice))

        return best_epoch, best_dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch-root', default='output/VOS')
    parser.add_argument('--gt-root', default='Youtube-VOS-CL/valid/Annotations')
    parser.add_argument('--best-log', default='output/best_log.txt')
    parser.add_argument('--result-log', default='output/result_log.txt')
    args = parser.parse_args()
    evaluate_VOS(args.epoch_root, args.gt_root, args.best_log, args.result_log).evaluate_all_epoches()
