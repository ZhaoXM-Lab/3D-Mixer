import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import auc
import logging
from .utils import DTYPE, eval_plot


def val_reg_epoch(global_step, val_loader, model, opt, device, writer):
    label_names = opt.label_names
    predicts, groundtruths, group_labels, val_loss = model.evaluate_data(val_loader, device, dtype=DTYPE)
    group_labels = np.array(list(map(lambda x: {0: 0, 1: 1, 2: 1, 3: 2, 4: -1, -1: -1}[x], group_labels.ravel())))

    metrix_strs = {}
    metrix_figs = {}
    for i in range(len(groundtruths[0])):
        pre = predicts[:, i, :]
        label = groundtruths[:, i, :]
        non_nan = ~np.isnan(label)
        pre = pre[non_nan]
        label = label[non_nan]

        metrix_strs['r_%s' % label_names[i]] = np.corrcoef(label, pre)[0, 1]
        metrix_strs['mae_%s' % label_names[i]] = np.mean(np.abs(pre - label))
        metrix_figs['y%d' % (i + 1)] = eval_plot(pre=pre, label=label, dis_label=group_labels[non_nan.ravel()],
                                                 vmin=0, vmax=2, show=False,
                                                 var_name=label_names[i])

    return metrix_figs, metrix_strs


def val_clf_epoch(global_step, val_loader, model, opt, device, writer):
    def evaluation(model, val_loader):
        if opt.framework == 'pytorch':
            predicts, groundtruths, group_labels, val_loss = model.evaluate_data(val_loader, device, dtype=DTYPE)
            try:
                val_loss = val_loss.detach().cpu().item()
            except:
                pass
            predict1 = predicts[:, 0, :]
            groundtruth1 = groundtruths[:, 0, :]
        else:
            raise NotImplementedError

        predict1 = np.array(predict1)
        groundtruth1 = np.array(groundtruth1)
        return predict1, groundtruth1, val_loss

    pre, label, val_loss = evaluation(model=model, val_loader=val_loader)
    pre[np.isnan(pre)] = 0
    prec, rec, thr = precision_recall_curve(label, pre)
    fpr, tpr, thr = roc_curve(label, pre)
    if auc(fpr, tpr) < (0.5 - 0.1):
        logging.warning('AUC is less than 0.5, reversed result is reported')
        prec, rec, thr = precision_recall_curve(label, -pre)
        fpr, tpr, thr = roc_curve(label, -pre)

    tn, fp, fn, tp = confusion_matrix(y_pred=pre.round(), y_true=label).ravel()

    metrix_strs = {}
    metrix_figs = {}
    metrix_strs['AUC'] = auc(fpr, tpr)
    metrix_strs['AUPR'] = auc(rec, prec)
    metrix_strs['ACC'] = accuracy_score(y_pred=pre.round(), y_true=label)
    metrix_strs['SEN'] = tp / (tp + fn)
    metrix_strs['SPE'] = tn / (tn + fp)
    metrix_strs['Val_Loss'] = val_loss

    return metrix_figs, metrix_strs


def validate(global_step, val_loader, net, opt, device, writer):
    if opt.clfsetting == 'regression':
        metrix_figs, metrix_strs = val_reg_epoch(global_step, val_loader, net, opt, device, writer)
        val_metrix = 0
        for label_name in opt.label_names:
            val_metrix -= metrix_strs['mae_%s' % label_name]
    else:
        metrix_figs, metrix_strs = val_clf_epoch(global_step, val_loader, net, opt, device, writer)
        val_metrix = -metrix_strs['Val_Loss']
    return metrix_figs, metrix_strs, val_metrix
