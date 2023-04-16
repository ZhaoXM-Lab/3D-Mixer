from utils.datasets import get_dataset
from utils.opts import parse_opts
from utils.utils import gen_center_mat
import numpy as np
import torch
import os
import random
from scipy.stats import ttest_ind
import pickle

seed = 1024

os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    num_pat = 50
    PATCH_SIZE = [25, 25, 25]

    opt = parse_opts()
    opt.dataset = 'ADNI_dx_BL'
    opt.clfsetting = 'CN-AD'
    opt.no_shift = True
    opt.flip_axises = None
    opt.center_mat = gen_center_mat(PATCH_SIZE)
    opt.patch_size = PATCH_SIZE
    opt.batch_size = 8

    train_loader, _, _ = get_dataset(dataset=opt.dataset, opt=opt)

    # t-test
    patches = []
    groups = []
    for n, data in enumerate(train_loader, 0):
        inputs, _, labels, _ = data
        groups.append(labels[:, 0, 0, 0])
        patches.append(inputs.mean(dim=[-4, -3, -2, -1]))

    patches = torch.cat(patches, dim=0)
    groups = torch.cat(groups, dim=0)

    patches = patches.numpy()
    groups = groups.numpy()

    assert (groups == 1).sum() < 0.5 * (groups.shape[0])
    ad_index = np.where(groups == 1)[0]
    cn_index = np.random.choice(np.where(groups == 0)[0], ad_index.shape[0])

    pvs = []
    for pat_ind in range(patches.shape[1]):
        a = patches[ad_index, pat_ind]
        b = patches[cn_index, pat_ind]
        s, p = ttest_ind(a, b, equal_var=False)
        pvs.append(p)

    pvs = np.array(pvs)
    diff_patinx = pvs.argsort()[:num_pat]
    DLADLMKS = opt.center_mat[:, diff_patinx]

    with open('./utils/DLADLMKS%d.pkl' % opt.patch_size[0], 'wb') as f:
        pickle.dump([DLADLMKS, opt.patch_size, diff_patinx], f)
