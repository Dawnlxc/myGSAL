import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from loader import check_mkdir, get_data
def evaluate_metrics(y_test, y_pred, return_mode='all'):
    n = y_pred.shape[0]
    all_accuracy = np.zeros(n)
    all_dice = np.zeros(n)
    all_jaccard = np.zeros(n)
    all_sensitivity = np.zeros(n)
    all_specificity = np.zeros(n)
    for i in range(n):
        gt, pred = y_test[i], y_pred[i]
        gt_flt = np.ndarray.flatten(gt)
        pred_flt = np.ndarray.flatten(pred)

        precisions, recalls, thresholds = precision_recall_curve(gt_flt, pred_flt)
        f1 = 2*(precisions * recalls) / (precisions + recalls)
        max_value = np.argmax(f1)
        precision, recall, thres = precisions[max_value], recalls[max_value], thresholds[max_value]

        pred_mask = (pred_flt > thres)
        pred_label = pred_mask*1

        tn, fp, fn, tp = confusion_matrix(gt_flt, pred_label).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        iou = tp / (tp + fp + fn)
        dice = 2*tp / (2*tp + fp + fn)
        specificity = tn / (tn + fp)

        all_accuracy[i] = accuracy
        all_dice[i] = dice
        all_jaccard[i] = iou
        all_sensitivity[i] = recall
        all_specificity[i] = specificity

    # print('Accuracy: {:4f}, Dice: {:4f}, Jaccard: {:4f}, Sensitivity: {:4f}, Specificity: {:4f}'.format(
    #     np.nanmean(all_accuracy), np.nanmean(all_dice), np.nanmean(all_jaccard), np.nanmean(all_sensitivity), np.nanmean(all_specificity)
    # ))
    if return_mode == 'all':
        return all_accuracy, all_dice, all_jaccard, all_sensitivity, all_specificity
    if return_mode == 'value':
        return np.nanmean(all_accuracy), np.nanmean(all_dice), np.nanmean(all_jaccard), np.nanmean(all_sensitivity), np.nanmean(all_specificity)
    else:
        return 'No val returned'


def evaluate(args, test_loader, img_size, G, device, smooth=1e-8):
    G.eval()
    n = len(test_loader)
    acc, dice, jcd, se, sp = 0.0, 0.0, 0.0, 0.0, 0.0
    # loss = 0.0
    save_path = os.path.join(args.root, 'results')
    check_mkdir(save_path)
    with torch.no_grad():
        for i, (img, gt) in enumerate(test_loader):
            img = img.to(device)
            gt = gt.to(device)

            _, _, seg = G(img)
            seg = nn.Sigmoid()(seg)

            gt_flt = torch.flatten(gt)
            pred_flt = torch.flatten(seg)

            precisions, recalls, thresholds = precision_recall_curve(gt_flt, pred_flt)
            f1 = 2*(precisions * recalls) / (precisions + recalls + smooth)
            max_value = np.argmax(f1)
            precision, recall, thres = precisions[max_value], recalls[max_value], thresholds[max_value]
            pred_mask = (pred_flt > thres)
            pred_label = pred_mask*1
            plt.imsave(os.path.join(save_path, f'{i}.png'), pred_label.cpu().numpy().reshape(img_size), cmap='gray')
            # plt.imshow(pred_label.cpu().numpy().reshape(img_size), cmap='gray')
            # plt.show()

            tn, fp, fn, tp = confusion_matrix(gt_flt, pred_label).ravel()
            acc += (tp + tn) / (tp + tn + fp + fn + smooth)
            jcd += tp / (tp + fp + fn + smooth)
            dice += 2*tp / (2*tp + fp + fn)
            se += recall
            sp += tn / (tn + fp + smooth)
    acc /= n
    dice /= n
    jcd /= n
    se /= n
    sp /= n
    # print('Acc: {:.4f}, Dice: {:.4f}, Jaccard: {:.4f}, SE: {:.4f}, SP: {:.4f}'.format(acc, dice, jcd, se, sp))
    return acc, dice, jcd, se, sp


model_path = '/Users/dawn/Desktop/GSAL/models/epoch99_acc0.9791_dice0.9533_jcd0.9107_se0.9359_sp0.9919.pth'

def test(args):
    X, y = get_data(args.root, img_size=args.img_size)

if __name__ == '__main__':
    evaluate(None, model_path)