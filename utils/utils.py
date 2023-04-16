import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import logging
from scipy.stats import norm
import pickle as pkl

DTYPE = torch.float32
BASEDIR = os.path.dirname(os.path.dirname(__file__))

NUM_LMKS = 50
with open(os.path.join(BASEDIR, 'utils/DLADLMKS25.pkl'), 'rb') as f:
    LMKS_CENTER_MAT, _, _ = pkl.load(f)
LMKS_CENTER_MAT = LMKS_CENTER_MAT[:, :NUM_LMKS]
LMKS_CORS = LMKS_CENTER_MAT.T
LMKS_ADJ = np.zeros([NUM_LMKS, NUM_LMKS])
for i in range(NUM_LMKS):
    for j in range(NUM_LMKS):
        if i > j:
            LMKS_ADJ[i, j] = np.linalg.norm(LMKS_CORS[i] - LMKS_CORS[j], ord=2)
LMKS_ADJ += LMKS_ADJ.T
LMKS_NEIGHBORS = []
for i in range(NUM_LMKS):
    LMKS_NEIGHBORS.append([[LMKS_ADJ[i].argsort()[:5]]])
LMKS_NEIGHBORS = np.array(LMKS_NEIGHBORS)

FULL_PATCH_SIZE = 117, 141, 117
IMG_SIZE = 121, 145, 121
FULL_CENTER_MAT = [[], [], []]
FULL_CENTER_MAT[0].append(int(np.floor((IMG_SIZE[0] - 1) / 2.0)))
FULL_CENTER_MAT[1].append(int(np.floor((IMG_SIZE[1] - 1) / 2.0)))
FULL_CENTER_MAT[2].append(int(np.floor((IMG_SIZE[2] - 1) / 2.0)))
FULL_CENTER_MAT = np.array(FULL_CENTER_MAT)


def mf_save_model(model, path, framework):
    if framework == 'pytorch':
        torch.save(model.state_dict(), path)
    elif framework == 'keras':
        model.save(path)
    else:
        raise NotImplementedError


def mf_load_model(model, path, framework, device='cpu'):
    if framework == 'pytorch':
        try:
            model.load_state_dict(torch.load(path, map_location=device), strict=True)
        except RuntimeError:
            model.load_state_dict(torch.load(path, map_location=device), strict=False)
            logging.warning('Loaded pretrain train model Unstrictly!')
    elif framework == 'keras':
        model.load_weights(path)
    else:
        raise NotImplementedError
    return model


def eval_plot(pre, label, dis_label, vmin, vmax, show=True, var_name=''):
    def line_plot(ind):
        c = ['r', 'g', 'coral'][ind]
        group = ['NC', 'MCI', 'AD'][ind]
        x, y = label[np.where(dis_label == ind)], pre[np.where(dis_label == ind)]
        plt.plot(x, y, '.', c=c, markersize=1.5)
        plt.plot([vmin, vmax], [vmin, vmax], c='k', linewidth=0.5)
        plt.title('%s: r=%.2f, MAE: %.2f' %
                  (group, np.corrcoef(x, y)[0, 1], np.mean(np.abs(x.flatten() - y.flatten()))))
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

    def line_plot_all():
        plt.plot(label[np.where(dis_label == 0)], pre[np.where(dis_label == 0)], '.',
                 c='r', markersize=1.5)
        plt.plot(label[np.where(dis_label == 1)], pre[np.where(dis_label == 1)], '.',
                 c='g', markersize=1.5)
        plt.plot(label[np.where(dis_label == 2)], pre[np.where(dis_label == 2)], '.',
                 c='coral', markersize=1.5)
        plt.plot([vmin, vmax], [vmin, vmax], c='k', linewidth=0.5)
        plt.title('ALL: r=%.2f, MAE: %.2f, r2: %.2f' %
                  (np.corrcoef(label, pre)[0, 1],
                   np.mean(np.abs(pre.flatten() - label.flatten())),
                   1 - np.sum((label - pre) ** 2.) / np.sum((label - label.mean()) ** 2.)))
        plt.legend(['NC', 'MCI', 'AD'])
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

    fig = plt.figure()
    plt.subplot(221)
    line_plot(0)
    plt.subplot(222)
    line_plot(1)
    plt.subplot(223)
    line_plot(2)
    plt.subplot(224)
    line_plot_all()
    plt.suptitle(var_name)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def gen_center_mat(pat_size: list):
    center_mat = [[], [], []]
    for x in np.arange(pat_size[0] // 2, IMG_SIZE[0] // pat_size[0] * pat_size[0], pat_size[0]):
        for y in np.arange(pat_size[1] // 2, IMG_SIZE[1] // pat_size[1] * pat_size[1], pat_size[1]):
            for z in np.arange(pat_size[2] // 2, IMG_SIZE[2] // pat_size[2] * pat_size[2], pat_size[2]):
                center_mat[0].append(x + (IMG_SIZE[0] % pat_size[0]) // 2)
                center_mat[1].append(y + (IMG_SIZE[1] % pat_size[1]) // 2)
                center_mat[2].append(z + (IMG_SIZE[2] % pat_size[2]) // 2)
    center_mat = np.array(center_mat)
    return center_mat


def num2vect(x, bin_range, bin_step, sigma):
    """adopted from https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        return np.array(x), np.array(1)
    else:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers


def count_params(model, framework='pytorch'):
    if framework == 'pytorch':
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
    elif framework == 'keras':
        params = model.count_params()
    else:
        raise NotImplementedError
    print('The network has {} params.'.format(params))


class Standr():
    def __init__(self, ori_data: np.ndarray, soft_label=False):
        non_nan = ~np.isnan(ori_data)
        self.non_nan = non_nan

        if non_nan.any():
            self.mean = ori_data[non_nan].mean()
            self.max = ori_data[non_nan].max()
            self.min = ori_data[non_nan].min()
            self.std = ori_data[non_nan].std()
            self._vec = soft_label

    def standr(self, ori_data):
        if not self.non_nan.any():
            return ori_data

        if not self._vec:
            return (ori_data - self.min) / (self.max - self.min)
        else:
            if self.max < 1000:  # for csf tau level
                vec_data, self.bin = num2vect(ori_data, [0, 1000], 25, sigma=20)
                return vec_data
            else:
                vec_data, self.bin = num2vect(ori_data, [-80, 4000], 102, sigma=80)
                return vec_data

    def unstandr(self, processed_data):
        if not self.non_nan.any():
            return processed_data
        if not self._vec:
            return processed_data * (self.max - self.min) + self.min
        else:
            return np.dot(processed_data, self.bin)
