import argparse
import pickle as pkl
from .utils import gen_center_mat
from .utils import FULL_CENTER_MAT, FULL_PATCH_SIZE, NUM_LMKS
import copy


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--result_path',
        default='results',
        type=str,
        help='Result directory path')

    parser.add_argument(
        '--dataset',
        default='ADNI_PET',
        type=str,
        help='Select dataset , MIRIAD|AIBL|OASIS|NACC is for testing only.'
             ' (ADNI_dx|ADNI_PETdx|ADNI_PET|MIRIAD|AIBL|OASIS|NACC)')

    parser.add_argument(
        '--label_names',
        default=['FDG', 'AV45'],
        type=str,
        nargs='+',
        help='regression lables (FDG, AV45)')

    parser.add_argument(
        '--modal',
        default='mwp1',
        type=str,
        help='modality of sMRI (mwp1|wm)')

    parser.add_argument(
        '--clfsetting',
        default='regression',
        type=str,
        help='classification setting (regression|CN-AD|sMCI-pMCI)')

    parser.add_argument(
        '--save_path',
        default=None,
        type=str,
        help='path to save model')

    parser.add_argument(
        '--method',
        default='RegMixer',
        type=str,
        help=
        'choose a method.'
    )
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='Batch Size')

    parser.add_argument(
        '--patch_size',
        default=(25, 25, 25),
        type=int,
        nargs=3,
        help='patch size, only available for some methods')

    parser.add_argument(
        '--val_as_train',
        action='store_true',
        help='If true, using validation set as training set')
    parser.set_defaults(val_as_train=False)

    parser.add_argument(
        '--val_as_test',
        action='store_true',
        help='If true, using validation set as testing set')
    parser.set_defaults(val_as_test=False)

    parser.add_argument(
        '--lr_mul_factor',
        default=0.5,
        type=float,
        help=
        'multiplication factor of learning rate scheduler')

    parser.add_argument(
        '--optimizer',
        default='adam',
        type=str,
        help=
        'Optimizer, adam|sgd')

    parser.add_argument(
        '--cuda_index',
        default=0,
        type=int,
        help='Specify the index of gpu')

    parser.add_argument(
        '--n_epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')

    parser.add_argument(
        '--learning_rate',
        default=1e-4,
        type=float,
        help=
        'Initial learning rate (divided by factor while training by lr scheduler)')

    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)

    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)

    parser.add_argument(
        '--no_log',
        action='store_true',
        help='If true, tensorboard logging is not used.')
    parser.set_defaults(no_log=False)

    parser.add_argument(
        '--n_threads',
        default=8,
        type=int,
        help='Number of threads for multi-thread loading')

    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )

    parser.add_argument(
        '--pretrain_path',
        default='',
        type=str,
        help='Pretrained model (.pth)')

    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)

    parser.add_argument(
        '--flip_axises',
        default=[0, 1, 2],
        type=int,
        nargs='+',
        help='flip axises (0, 1, 2)')

    parser.add_argument(
        '--no_smooth',
        action='store_true',
        help='no smooth apply to MRI')
    parser.set_defaults(no_smooth=False)

    parser.add_argument(
        '--no_shift',
        action='store_true',
        help='no shift apply to MRI for data augmentation')
    parser.set_defaults(no_shift=False)

    parser.add_argument(
        '--no_shuffle',
        action='store_true',
        help='no shuffle apply to batch sampling')
    parser.set_defaults(no_shuffle=False)

    parser.add_argument(
        '--method_para',
        default='{}',
        type=str,
        help='specify method parameters in dict form. eg: {"para1": values1}')

    args = parser.parse_args()
    return args


def mod_opt(method, opt):
    opt = copy.copy(opt)
    if method in ['Res18']:
        opt.center_mat = FULL_CENTER_MAT
        opt.patch_size = FULL_PATCH_SIZE
        opt.method_setting = {'sample_size': opt.patch_size, 'num_classes': opt.num_classes}
    elif method in ['LDMIL', 'LDMIL_BN']:
        with open('./utils/DLADLMKS%d.pkl' % opt.patch_size[0], 'rb') as f:
            opt.center_mat, opt.patch_size, _ = pkl.load(f)
        opt.center_mat = opt.center_mat[:, :NUM_LMKS]

        if not opt.patch_size[0] == 25:
            opt.resample_patch = [25, 25, 25]
        opt.method_setting = {
            'num_patch': opt.center_mat.shape[1], 'patch_size': opt.patch_size, 'inplanes': 1}
    elif method in ['ClfMixer', 'RegMixer', 'FuseMixer']:
        opt.center_mat = gen_center_mat(opt.patch_size)
        opt.method_setting = {
            'patch_size': opt.patch_size, 'num_patches': opt.center_mat.shape[1],
            'dim': 64, 'depth': 4, 'dropout': 0.1
        }

        if method == 'ClfMixer':
            with open('./utils/DLADLMKS%d.pkl' % opt.patch_size[0], 'rb') as f:
                opt.center_mat, opt.patch_size, _ = pkl.load(f)
                opt.center_mat = opt.center_mat[:, :NUM_LMKS]
            opt.method_setting['num_patches'] = opt.center_mat.shape[1]

        elif method == 'FuseMixer':
            with open('./utils/DLADLMKS%d.pkl' % opt.patch_size[0], 'rb') as f:
                _, _, patinx = pkl.load(f)
                patinx = patinx[:NUM_LMKS]
                opt.method_setting['dropout'] = 0.3
            opt.method_setting['clfpat_index'] = patinx

        elif method == 'RegMixer':
            opt.method_setting = {
                'patch_size': opt.patch_size, 'num_patches': opt.center_mat.shape[1],
                'dim': 64, 'depth': 4, 'dropout': 0.1, 'num_regmixer': 1
            }

    elif method == 'DAMIDL':
        with open('./utils/DLADLMKS%d.pkl' % opt.patch_size[0], 'rb') as f:
            opt.center_mat, opt.patch_size, _ = pkl.load(f)

        opt.center_mat = opt.center_mat[:, :NUM_LMKS]
        if not opt.patch_size[0] == 25:
            opt.resample_patch = [25, 25, 25]
        opt.method_setting = {
            'patch_num': opt.center_mat.shape[1], 'feature_depth': [32, 64, 128, 128]
        }
    elif method == 'ViT':
        opt.center_mat = gen_center_mat(opt.patch_size)
        opt.method_setting = {
            'patch_size': opt.patch_size, 'num_patches': opt.center_mat.shape[1],
            'dim': 64, 'depth': 4, 'heads': 8, 'dim_head': 64, 'mlp_dim': 4 * 64,
            'dropout': 0.1, 'emb_dropout': 0.1, 'num_classes': 1, 'pool': 'cls', 'channels': 1}
    else:
        raise NotImplementedError

    return opt
