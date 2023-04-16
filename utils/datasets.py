import copy
import logging

logging.basicConfig(level='WARNING')
import torch
import numpy as np
import os
import torch.utils.data as data
from utils.utils import Standr, DTYPE, BASEDIR
import pickle as pkl
from nilearn.image import smooth_img
import nibabel as nib
import pandas as pd
from scipy import ndimage as nd

ADNI_PATH = '/data/datasets/ADNI/ADNI_T1'


def batch_sampling(imgs, labels, center_mat, aux_labels, dis_labels, patch_size=(25, 25, 25), random=False,
                   shift=False, flip_axis=None):
    shift_range = [-2, -1, 0, 1, 2]
    flip_pro = 0.3
    num_patch = len(center_mat[0])
    batch_size = len(imgs)
    margin = [int(np.floor((i - 1) / 2.0)) for i in patch_size]

    batch_img = torch.tensor(data=np.zeros([num_patch * batch_size, 1, patch_size[0], patch_size[1], patch_size[2]]),
                             dtype=DTYPE)
    batch_label = torch.tensor(data=np.zeros([num_patch * batch_size] + list(labels.shape[1:])), dtype=DTYPE)
    batch_aux_label = torch.tensor(data=np.zeros([num_patch * batch_size] + list(aux_labels.shape[1:])), dtype=DTYPE)
    batch_dis_label = torch.tensor(data=np.zeros([num_patch * batch_size] + list(dis_labels.shape[1:])), dtype=DTYPE)

    for num, data in enumerate(zip(imgs, labels, aux_labels, dis_labels), start=0):
        img, label, aux_label, dis_label = data
        if not random:
            for ind, cors in enumerate(zip(center_mat[0], center_mat[1], center_mat[2])):
                x_cor, y_cor, z_cor = cors
                if shift:
                    x_scor = x_cor + shift_range[torch.randint(high=len(shift_range), size=(1,))]
                    y_scor = y_cor + shift_range[torch.randint(high=len(shift_range), size=(1,))]
                    z_scor = z_cor + shift_range[torch.randint(high=len(shift_range), size=(1,))]
                else:
                    x_scor, y_scor, z_scor = x_cor, y_cor, z_cor

                single_patch = img[:,
                               max(x_scor - margin[0], 0): x_scor + margin[0] + 1,
                               max(y_scor - margin[1], 0): y_scor + margin[1] + 1,
                               max(z_scor - margin[2], 0): z_scor + margin[2] + 1]
                if (not (flip_axis is None)) and (torch.rand(1) < flip_pro):
                    if isinstance(flip_axis, list):
                        single_patch = single_patch.flip(flip_axis[torch.randint(high=len(flip_axis), size=(1,))])
                    single_patch = single_patch.flip(flip_axis)

                batch_img[ind + num * num_patch,
                :single_patch.shape[0],
                :single_patch.shape[1],
                :single_patch.shape[2],
                :single_patch.shape[3]] = single_patch

                # batch_img[ind + num * num_patch] = single_patch

                batch_label[ind + num * num_patch] = label
                batch_aux_label[ind + num * num_patch] = aux_label
                batch_dis_label[ind + num * num_patch] = dis_label
        else:
            raise NotImplementedError

    return batch_img, batch_aux_label, batch_label, batch_dis_label


class Dataset(object):
    def __init__(self):
        self.preload = False
        self.data = []
        self.namelist = []
        self.maskmat = None
        self.image_path = ''
        self.smooth = False
        self.labels = [[]]
        self.aux_labels = None
        self.dis_label = None
        self.standr_p1 = None
        self.standr_p2 = None
        self.standrs = []
        self._id_info = []

    @staticmethod
    def _filtering(info):
        '''filtering for AD->CN

        Args:
            info:

        Returns:

        '''
        p_rid = []
        _info = info.copy()[['RID', 'VISCODE', 'NEW_DX']]
        _info.loc[_info['VISCODE'] == 'bl', 'VISCODE'] = 0
        _info['VISCODE'] = _info['VISCODE'].apply(
            func=lambda x: int(x.replace('m', '')) if isinstance(x, str) else x)
        _info['NEW_DX'] = _info['NEW_DX'].astype(str).apply(
            func=lambda x: {'CN': 0, 'sMCI': 0, 'pMCI': 1, 'AD': 1, 'nan': -1}[x])

        _info = _info.sort_values(by=['RID', 'VISCODE'], kind='mergesort')
        _info = _info[_info['NEW_DX'] != -1]
        for i in _info['RID'].drop_duplicates().values:
            temp = _info[_info['RID'] == i]
            if not (
                    temp.sort_values(by=['VISCODE'], kind='mergesort')['NEW_DX'].values ==
                    temp.sort_values(by=['NEW_DX'],
                                     kind='mergesort')['NEW_DX'].values).all():
                p_rid.append(i)
        # info[info['RID'].isin(p_rid)].iloc[:, :5].sort_values(by=['RID', 'VISCODE'])
        return p_rid

    def get_id_info(self):
        if len(self._id_info):
            return self._id_info
        else:
            raise NotImplementedError

    def load_data(self, img_path, name, smooth=False):
        if 'NATIVE_GM_' in name:
            dir, file = os.path.split(os.path.join(img_path, name))
            file = file.replace('NATIVE_GM_', '')
            ori_img = nib.load(os.path.join(dir, '../', file))
            brain_label = nib.load(os.path.join(dir, 'p0' + file)).get_fdata()
        elif 'NATIVE_WM_' in name:
            dir, file = os.path.split(os.path.join(img_path, name))
            file = file.replace('NATIVE_GM_', '')
            ori_img = nib.load(os.path.join(dir, '../', file))
            brain_label = nib.load(os.path.join(dir, 'p0' + file)).get_fdata()
        else:
            ori_img = nib.load(os.path.join(img_path, name))

        if smooth:
            ori_img = smooth_img(ori_img, 8).get_fdata()
        else:
            ori_img = ori_img.get_fdata()

        if 'NATIVE_GM_' in name:
            ori_img[brain_label < 1.5] = 0
            ori_img[brain_label >= 2.5] = 0
        elif 'NATIVE_WM_' in name:
            ori_img[brain_label < 2.5] = 0

        ori_img[np.isnan(ori_img)] = 0
        return ori_img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.preload:
            if self.data[index] is not None:
                bat_data = self.data[index]
            else:
                name = self.namelist[index]
                if self.maskmat is None:
                    bat_data = (self.load_data(self.image_path, name, self.smooth))
                else:
                    bat_data = (self.load_data(self.image_path, name, self.smooth) * self.maskmat)
                bat_data = np.array(bat_data, dtype=np.float32)
                bat_data[np.isnan(bat_data)] = 0
                self.data[index] = bat_data
        else:
            name = self.namelist[index]
            if self.maskmat is None:
                bat_data = (self.load_data(self.image_path, name, self.smooth))
            else:
                bat_data = (self.load_data(self.image_path, name, self.smooth) * self.maskmat)
            bat_data = np.array(bat_data, dtype=np.float32)
            bat_data[np.isnan(bat_data)] = 0

        bat_data = torch.from_numpy(bat_data).unsqueeze(0).unsqueeze(0)  # channel, batch
        bat_labels = torch.Tensor([self.labels[index]])
        if self.aux_labels is not None:
            bat_aux_label = torch.Tensor([self.aux_labels[index]])
        else:
            bat_aux_label = bat_labels * torch.from_numpy(np.array([np.nan]))

        if self.dis_label is not None:
            bat_dis_label = torch.Tensor([self.dis_label[index]])
        else:
            bat_dis_label = torch.from_numpy(np.array([np.nan]))

        return bat_data, bat_labels, bat_aux_label, bat_dis_label

    def __len__(self):
        return len(self.namelist)


class Patch_Data(data.Dataset):
    def __init__(self, imgdata, patch_size, center_mat, shift, flip_axis, resample_patch=None):
        '''

        Args:
            imgdata:
            patch_size:
            center_mat:
            shift:
            flip_axis:
            resample_patch: list of resample shape eg. [25, 25, 25]
        '''
        self.patch_size = patch_size
        self.center_mat = center_mat
        self.shift = shift
        self.flip_axis = flip_axis
        self.imgdata = imgdata
        self.standrs = imgdata.standrs
        self.standr_p1 = imgdata.standr_p1
        self.standr_p2 = imgdata.standr_p2
        self.labels = imgdata.labels
        self.resample_patch = resample_patch

    def __getitem__(self, index):
        bat_data, bat_labels, bat_aux_label, bat_dis_label = self.imgdata[index]
        inputs, aux_labels, labels, dis_label = batch_sampling(imgs=bat_data, labels=bat_labels,
                                                               center_mat=self.center_mat,
                                                               aux_labels=bat_aux_label, dis_labels=bat_dis_label,
                                                               patch_size=self.patch_size,
                                                               random=False, shift=self.shift,
                                                               flip_axis=self.flip_axis,
                                                               )

        if self.resample_patch is not None:
            assert len(self.resample_patch) == 3
            _inputs = inputs.numpy()
            resam = np.zeros(list(inputs.shape[:-3]) + self.resample_patch)
            dsfactor = [w / float(f) for w, f in zip(self.resample_patch, _inputs.shape[-3:])]
            for i in range(_inputs.shape[0]):
                resam[i, 0, :] = nd.interpolation.zoom(_inputs[i, 0, :], zoom=dsfactor)
            inputs = torch.from_numpy(resam)

        return inputs.squeeze(0), aux_labels.squeeze(0), labels.squeeze(0), dis_label.squeeze(0),

    def __len__(self):
        return len(self.imgdata)


def train_test_nooverlap(data_size, test_size, id_info, seed=None):
    if not (seed is None):
        np.random.seed(seed)
    test = []
    ids_set = list(set(id_info))

    inxes = np.arange(data_size)
    ids_set.sort()  # remove randomness induced by former step
    np.random.shuffle(ids_set)

    for id in ids_set:
        test += (inxes[id_info == id]).tolist()
        if (np.unique(id_info[test]).shape[0] / np.unique(id_info).shape[0]) > test_size:
            break
    train = list(set(inxes).difference(set(test)))
    train = sorted(train)
    return train, test


class ADNI(Dataset):
    def __init__(self, subset, cohort, label_names: list, opt, only_bl=False, ids_exclude=(), ids_include=None):
        '''

        Args:
            subset: list or string： training|validation|testing
            cohort: ADNI1|ADNI1-P|ADNI2|ADNI2-P
            fix_imbalance:
            label_names:
            clfsetting:
            modal:
            ids_exclude:
        '''
        super(ADNI, self).__init__()

        assert cohort in ['ADNI1', 'ADNI2', 'ADNI1-P', 'ADNI2-P', 'PET', 'NO-PET', 'ALL']

        clfsetting = opt.clfsetting
        modal = opt.modal
        self.image_path = ADNI_PATH
        self.smooth = not opt.no_smooth
        self.maskmat = opt.mask_mat
        self.standr_p1 = None
        self.standr_p2 = None
        self.standrs = []

        LABEL = []
        [DX, FDG, AV45], DIS_LABEL, NAME_LIST, id_info, info = self.get_data(self.image_path, modal,
                                                                             ids_exclude)

        if clfsetting == 'CN-AD':
            DX = np.array(
                list(map(lambda x: {'CN': 0, 'sMCI': -1, 'pMCI': -1, 'MCI': -1, 'AD': 1, 'nan': -1}[str(x)], DX)))
        elif clfsetting == 'sMCI-pMCI':
            DX = np.array(list(map(lambda x: {'CN': -1, 'sMCI': 0, 'pMCI': 1, 'AD': -1, 'nan': -1}[str(x)], DX)))
        elif clfsetting == 'regression':
            DX = np.array(list(map(lambda x: {'CN': 0, 'sMCI': 1, 'pMCI': 2, 'AD': 3, 'nan': 4}[str(x)], DX)))

        DX = DX.reshape(-1, 1)
        if 'DX' in label_names:
            LABEL.append(DX)
            assert len(label_names) == 1
        if 'FDG' in label_names:
            LABEL.append(FDG[:, -1:])
        if 'AV45' in label_names:
            LABEL.append(AV45[:, -1:])

        LABEL = np.array(list(zip(*LABEL)))

        if (('FDG' in label_names) or ('AV45' in label_names) or (opt.clfsetting == 'regression')):
            self.standrs = []
            for i in range(LABEL.shape[1]):
                s = Standr(LABEL[:, i, :], soft_label=False)
                self.standrs.append(s)
                LABEL[:, i, :] = s.standr(ori_data=LABEL[:, i, :])

            # for backward compatible
            try:
                self.standr_p1 = self.standrs[0]
                self.standr_p2 = self.standrs[1]
            except:
                pass

        inx = ~np.isnan(LABEL).all(axis=-1).all(axis=-1)

        # *note*: if a subject got several DX in diff time point, then only use the earliest DX
        for i in np.sort(np.unique(id_info)):
            if DX[id_info == i][0] == -1:
                inx[id_info == i] = False
        inx &= (DX[:, 0] != -1)

        # to garantee no overlapping
        adni1_subs = info[info['COLPROT'] == 'ADNI1']['RID'].drop_duplicates().values
        adni2_subs = info[info['COLPROT'] == 'ADNI2']['RID'].drop_duplicates().values
        pet_subs = info[~np.isnan(np.concatenate([FDG, AV45], axis=-1)).all(axis=1)]['RID'].drop_duplicates().values
        if cohort in ['ADNI1', 'ADNI2']:
            if cohort == 'ADNI1':
                inx &= info['COLPROT'] == cohort
            else:
                inx &= (info['COLPROT'] == cohort) & (~info['RID'].isin(adni1_subs))
        elif cohort in ['ADNI1-P', 'ADNI2-P']:
            if cohort == 'ADNI1-P':
                inx &= info['RID'].isin(adni1_subs)
            else:
                inx &= info['RID'].isin(adni2_subs) & (~info['RID'].isin(adni1_subs))
        elif cohort == 'PET':
            inx &= ~np.isnan(np.concatenate([FDG, AV45], axis=-1)).all(axis=1)
        elif cohort == 'NO-PET':
            inx &= ~info['RID'].isin(pet_subs)
        elif cohort == 'ALL':
            pass
        else:
            raise NotImplementedError

        LABEL = LABEL[inx]
        DIS_LABEL = DIS_LABEL[inx]
        NAME_LIST = NAME_LIST[inx]
        id_info = id_info[inx]
        info = info[inx]
        DX = DX[inx]

        # data spliting
        TRAIN_INX, TEST_VAL_INX = train_test_nooverlap(len(NAME_LIST), test_size=0.3, id_info=id_info, seed=7758)
        VAL_INX, TEST_INX = train_test_nooverlap(len(TEST_VAL_INX), test_size=0.5,
                                                 id_info=id_info[TEST_VAL_INX], seed=7758)
        TEST_INX = np.array(TEST_VAL_INX)[TEST_INX].tolist()
        VAL_INX = np.array(TEST_VAL_INX)[VAL_INX].tolist()

        ind = []

        if ids_include is None:
            if not isinstance(subset, list):
                subset = [subset]
            for ss in subset:
                if ss == 'training':
                    ind += TRAIN_INX
                elif ss == 'validation':
                    ind += VAL_INX
                elif ss == 'testing':
                    ind += TEST_INX
        else:
            ind = np.arange(len(NAME_LIST))[np.isin(id_info, ids_include)]

        ind = np.array(ind)
        # ind = ind[DX.ravel()[ind] != -1]
        if only_bl:
            new_ind = []
            ind = np.array(ind)
            temp = np.array([os.path.split(p)[-1].replace(modal, '') for p in NAME_LIST[ind]])
            temp_id = id_info[ind]
            for name in sorted(set(temp_id)):
                vis, visind = sorted(zip(temp[temp_id == name], ind[temp_id == name]), key=lambda x: x[0])[0]
                new_ind.append(visind)
            ind = new_ind

        self.id_info = id_info[ind]
        self._id_info = id_info[ind]
        self.info = info.iloc[ind, :]
        self.labels = LABEL[ind]

        self.dis_label = DIS_LABEL[ind]
        self.namelist = NAME_LIST[ind]

        self.preload = True
        if self.preload:
            self.data = [None for i in self.namelist]

    @staticmethod
    def get_fdg():
        UCB_FDG = pd.read_csv(os.path.join(BASEDIR, 'data/UCBERKELEYFDG_05_28_20.csv'), dtype=str)
        UCB_FDG.rename(columns={'EXAMDATE': 'EXAMDATE_FDG', }, inplace=True)
        rois = [["Angular", "Left"],
                ["Angular", "Right"],
                ["CingulumPost", "Bilateral"],
                ["Temporal", "Left"],
                ["Temporal", "Right"], ]
        ROI_FDG = []
        UCB_FDG = UCB_FDG[~pd.isnull(UCB_FDG['VISCODE2'])]

        # problematic data(viscode is wrong and duplicated)
        UCB_FDG = UCB_FDG[UCB_FDG['RID'] != '4765']
        UCB_FDG = UCB_FDG.sort_values(by=['RID', 'VISCODE2'], kind='mergesort')
        ind_df = UCB_FDG[['RID', 'VISCODE2', 'EXAMDATE_FDG']].drop_duplicates()
        for roi, l in rois:
            temp = UCB_FDG[(UCB_FDG['ROINAME'] == roi) & (UCB_FDG['ROILAT'] == l)]
            ROI_FDG.append(temp['MEAN'].values)
            assert (temp[['RID', 'VISCODE2']].values == ind_df[['RID', 'VISCODE2']].values).all()
        ROI_FDG = pd.DataFrame(np.array(ROI_FDG).T, columns=['_'.join(i) for i in rois])
        ROI_FDG['RID'] = ind_df.values[:, 0]
        ROI_FDG['VISCODE'] = ind_df.values[:, 1]
        ROI_FDG['FDG_SUMMARY'] = ROI_FDG[['_'.join(i) for i in rois]].astype(float).values.mean(axis=1)
        return ROI_FDG

    @staticmethod
    def get_av45():
        UCB_AV45 = pd.read_csv(os.path.join(BASEDIR, 'data/UCBERKELEYAV45_01_14_21.csv'), dtype=str)
        UCB_AV45 = UCB_AV45.drop('VISCODE', axis=1)
        UCB_AV45 = UCB_AV45.drop('update_stamp', axis=1)
        UCB_AV45.rename(columns={'EXAMDATE': 'EXAMDATE_AV45', 'VISCODE2': 'VISCODE'}, inplace=True)

        ROI_AV45 = UCB_AV45[['RID', 'VISCODE', 'EXAMDATE_AV45', 'FRONTAL_SUVR', 'CINGULATE_SUVR', 'PARIETAL_SUVR',
                             'TEMPORAL_SUVR', 'COMPOSITE_REF_SUVR', 'SUMMARYSUVR_COMPOSITE_REFNORM',
                             'SUMMARYSUVR_COMPOSITE_REFNORM_0.78CUTOFF']]

        return ROI_AV45

    @staticmethod
    def get_info():
        info = pd.read_csv(os.path.join(BASEDIR, 'data/ADNIMERGE_fixed.csv'), dtype=str)
        info.loc[(info['RID'] == '2') & (info['VISCODE'] == 'm90'), 'EXAMDATE'] = '2013/3/2'
        return info[['RID', 'VISCODE', 'COLPROT', 'EXAMDATE', ]]

    @staticmethod
    def get_dx():
        info = pd.read_csv(os.path.join(BASEDIR, 'data/ADNIMERGE_fixed.csv'), dtype=str)
        ids = set(info['RID'].to_list())
        ids = list(ids)
        ids.sort()

        info_ = info.copy()
        info_.loc[info_['DX_bl'] == 'EMCI', 'DX_bl'] = 'MCI'
        info_.loc[info_['DX_bl'] == 'LMCI', 'DX_bl'] = 'MCI'
        info_.loc[info_['DX_bl'] == 'SMC', 'DX_bl'] = 'CN'
        info_.loc[info_['DX'] == 'Dementia', 'DX'] = 'AD'

        for name in ids:
            inds = info_.loc[info_['RID'] == name, ['VISCODE', 'DX']].sort_values(by='VISCODE').index
            for i, ind in enumerate(inds[:-1]):
                if info_.loc[inds[i]]['DX'] is np.nan:
                    if i == 0:
                        if info_.loc[inds[i + 1]]['DX'] == 'CN':
                            info_.loc[inds[i], 'DX'] = 'CN'
                    elif ((info_.loc[inds[i - 1]]['DX'] is not np.nan) and
                          (info_.loc[inds[i - 1]]['DX'] == info_.loc[inds[i + 1]]['DX'])):
                        info_.loc[inds[i], 'DX'] = info_.loc[inds[i + 1], 'DX']

        for i in info_.index:
            if info_.loc[i]['DX'] == 'CN':
                info_.loc[i, 'NEW_DX'] = 'CN'
            elif info_.loc[i]['DX'] == 'AD':
                info_.loc[i, 'NEW_DX'] = 'AD'
            elif info_.loc[i]['DX'] == 'MCI':
                rid = info_.loc[i]['RID']
                dxs = info_[info_['RID'] == rid]['DX'].values
                if 'AD' in dxs:
                    info_.loc[i, 'NEW_DX'] = 'pMCI'
                else:
                    info_.loc[i, 'NEW_DX'] = 'sMCI'

        assert info_.loc[:, 'NEW_DX'].count() == info_.loc[:, 'DX'].count()
        info.insert(loc=info.columns.to_list().index('DX') + 1, column='NEW_DX', value=info_['NEW_DX'])

        info = info[['RID', 'VISCODE', 'NEW_DX']]
        return info

    def get_data(self, image_path, modal='mwp1', ids_exclude=()):
        info = self.get_info()
        FDG_INFO = self.get_fdg()
        AV45_INFO = self.get_av45()
        DX_INFO = self.get_dx()

        info = info[~info['RID'].isin(ids_exclude)]
        info['IMAGE'] = info[['RID', 'VISCODE']].apply(axis=1, func=lambda x: os.path.exists(
            os.path.join(image_path, x['RID'], 'report', 'catreport_' + x['VISCODE'] + '.pdf')))
        info = info[info['IMAGE']]

        for df in [DX_INFO, FDG_INFO, AV45_INFO]:
            info = info.merge(df, how='left', on=['RID', 'VISCODE'], )

        info = info[~info.iloc[:, 4:-7].isnull().all(axis=1)]
        info = info[~info['RID'].isin(self._filtering(info))]

        IND_INFO = info[['RID', 'VISCODE']].values
        id_info = info['RID'].values
        NAME_LIST = np.array([os.path.join(i[0], 'mri', modal + i[-1] + '.nii') for i in IND_INFO])
        DIS_LABEL = info['NEW_DX'].astype(str).apply(func=lambda x: {'CN': 0, 'sMCI': 1,
                                                                     'pMCI': 2, 'AD': 3, 'nan': 4}[x]).values

        DX = info['NEW_DX'].values
        FDG = info[['Angular_Left', 'Angular_Right', 'CingulumPost_Bilateral',
                    'Temporal_Left', 'Temporal_Right', 'FDG_SUMMARY']].astype(float).values
        AV45 = info[['FRONTAL_SUVR', 'CINGULATE_SUVR', 'PARIETAL_SUVR',
                     'TEMPORAL_SUVR', 'COMPOSITE_REF_SUVR',
                     'SUMMARYSUVR_COMPOSITE_REFNORM', ]].astype(float).values

        return [DX, FDG, AV45], DIS_LABEL, NAME_LIST, id_info, info


def get_dataset(dataset, opt, resample_patch=None):
    if dataset == 'ADNI_PET':
        data_train = ADNI(subset=['training', ], cohort='PET', label_names=['FDG', 'AV45'], opt=opt, ids_exclude=())
        data_val = ADNI(subset=['validation'], cohort='PET', label_names=['FDG', 'AV45'], opt=opt, ids_exclude=(),
                        only_bl=True)
        data_test = ADNI(subset=['testing'], cohort='PET', label_names=['FDG', 'AV45'], opt=opt, ids_exclude=(),
                         only_bl=True)

    elif dataset == 'ADNI_dx':
        data_train = ADNI(subset=['training', 'testing'], cohort='ADNI1-P', label_names=['DX'], opt=opt, ids_exclude=())
        data_val = ADNI(subset=['validation'], cohort='ADNI1-P', label_names=['DX'], opt=opt, ids_exclude=()
                        , only_bl=True)
        data_test = ADNI(subset=['training', 'validation', 'testing'], cohort='ADNI2-P', label_names=['DX'], opt=opt,
                         ids_exclude=(), only_bl=True)

    elif dataset == 'ADNI_dx_BL':
        data_train = ADNI(subset=['training', 'testing'], cohort='ADNI1-P', label_names=['DX'], opt=opt, ids_exclude=(),
                          only_bl=True)
        data_val = ADNI(subset=['validation'], cohort='ADNI1-P', label_names=['DX'], opt=opt, ids_exclude=(),
                        only_bl=True)
        data_test = ADNI(subset=['training', 'validation', 'testing'], cohort='ADNI2-P', label_names=['DX'], opt=opt,
                         ids_exclude=(), only_bl=True)


    elif dataset == 'ADNI_PETdx':  # each set (train, val, test) is the super set of ADNI_PET
        _regopt = copy.copy(opt)
        _regopt.clfsetting = 'regression'
        pet_train = ADNI(subset=['training', ], cohort='PET', label_names=['FDG', 'AV45'], opt=_regopt, ids_exclude=())
        pet_val = ADNI(subset=['validation'], cohort='PET', label_names=['FDG', 'AV45'], opt=_regopt, ids_exclude=(),
                       only_bl=True)
        pet_test = ADNI(subset=['testing'], cohort='PET', label_names=['FDG', 'AV45'], opt=_regopt, ids_exclude=(),
                        only_bl=True)
        pet_ids = np.unique(np.concatenate([pet_train.id_info, pet_val.id_info, pet_test.id_info]))

        dx_train = ADNI(subset=['training'], cohort='ALL', label_names=['DX'], opt=opt,
                        ids_exclude=pet_ids)
        dx_val = ADNI(subset=['validation'], cohort='ALL', label_names=['DX'], opt=opt,
                      ids_exclude=pet_ids, only_bl=True)
        dx_test = ADNI(subset=['testing'], cohort='ALL', label_names=['DX'], opt=opt,
                       ids_exclude=pet_ids, only_bl=True)

        data_train = ADNI(subset=None, cohort='ALL', label_names=['DX'], opt=opt,
                          ids_exclude=(), ids_include=np.unique(np.concatenate([pet_train.id_info, dx_train.id_info])))
        data_val = ADNI(subset=None, cohort='ALL', label_names=['DX'], opt=opt,
                        ids_exclude=(), ids_include=np.unique(np.concatenate([pet_val.id_info, dx_val.id_info])),
                        only_bl=True)
        data_test = ADNI(subset=None, cohort='ALL', label_names=['DX'], opt=opt, ids_exclude=(),
                         ids_include=np.unique(np.concatenate([pet_test.id_info, dx_test.id_info])),
                         only_bl=True)

    elif dataset == 'ADNI_PETdx_BL':  # each set (train, val, test) is the super set of ADNI_PET
        _regopt = copy.copy(opt)
        _regopt.clfsetting = 'regression'
        pet_train = ADNI(subset=['training', ], cohort='PET', label_names=['FDG', 'AV45'], opt=_regopt, ids_exclude=())
        pet_val = ADNI(subset=['validation'], cohort='PET', label_names=['FDG', 'AV45'], opt=_regopt, ids_exclude=(),
                       only_bl=True)
        pet_test = ADNI(subset=['testing'], cohort='PET', label_names=['FDG', 'AV45'], opt=_regopt, ids_exclude=(),
                        only_bl=True)
        pet_ids = np.unique(np.concatenate([pet_train.id_info, pet_val.id_info, pet_test.id_info]))

        dx_train = ADNI(subset=['training'], cohort='ALL', label_names=['DX'], opt=opt,
                        ids_exclude=pet_ids)
        dx_val = ADNI(subset=['validation'], cohort='ALL', label_names=['DX'], opt=opt,
                      ids_exclude=pet_ids, only_bl=True)
        dx_test = ADNI(subset=['testing'], cohort='ALL', label_names=['DX'], opt=opt,
                       ids_exclude=pet_ids, only_bl=True)

        data_train = ADNI(subset=None, cohort='ALL', label_names=['DX'], opt=opt,
                          ids_exclude=(), ids_include=np.unique(np.concatenate([pet_train.id_info, dx_train.id_info])),
                          only_bl=True)
        data_val = ADNI(subset=None, cohort='ALL', label_names=['DX'], opt=opt,
                        ids_exclude=(), ids_include=np.unique(np.concatenate([pet_val.id_info, dx_val.id_info])),
                        only_bl=True)
        data_test = ADNI(subset=None, cohort='ALL', label_names=['DX'], opt=opt, ids_exclude=(),
                         ids_include=np.unique(np.concatenate([pet_test.id_info, dx_test.id_info])),
                         only_bl=True)

    else:
        raise NotImplementedError

    if data_train:
        data_train = Patch_Data(data_train, patch_size=opt.patch_size, center_mat=opt.center_mat,
                                shift=not opt.no_shift, flip_axis=opt.flip_axises, resample_patch=resample_patch)
        data_train = torch.utils.data.DataLoader(data_train, batch_size=opt.batch_size, shuffle=not opt.no_shuffle,
                                                 num_workers=opt.n_threads, pin_memory=True)
    if data_val:
        data_val = Patch_Data(data_val, patch_size=opt.patch_size, center_mat=opt.center_mat,
                              shift=False, flip_axis=None, resample_patch=resample_patch)
        data_val = torch.utils.data.DataLoader(data_val, batch_size=opt.batch_size, shuffle=False,
                                               num_workers=opt.n_threads, pin_memory=True)
    data_test = Patch_Data(data_test, patch_size=opt.patch_size, center_mat=opt.center_mat,
                           shift=False, flip_axis=None, resample_patch=resample_patch)
    data_test = torch.utils.data.DataLoader(data_test, batch_size=opt.batch_size, shuffle=False,
                                            num_workers=opt.n_threads, pin_memory=True)

    if dataset not in ['MIRIAD', 'OASIS', 'AIBL', 'NACC', ] and not opt.val_as_train and not opt.val_as_test:
        checkpath = os.path.join(BASEDIR, 'utils', 'datacheck', dataset + '_' + opt.clfsetting + '.pkl')
        if os.path.exists(checkpath):
            with open(checkpath, 'rb') as f:
                id_infos = pkl.load(f)
            assert (id_infos[0] == data_train.dataset.imgdata.id_info).all()
            assert (id_infos[1] == data_val.dataset.imgdata.id_info).all()
            assert (id_infos[2] == data_test.dataset.imgdata.id_info).all()
        else:
            with open(checkpath, 'wb') as f:
                pkl.dump([data_train.dataset.imgdata.id_info,
                          data_val.dataset.imgdata.id_info,
                          data_test.dataset.imgdata.id_info], f)

        a = set(data_train.dataset.imgdata.id_info)
        b = set(data_val.dataset.imgdata.id_info)
        c = set(data_test.dataset.imgdata.id_info)
        assert len(a.intersection(b)) == len(a.intersection(c)) == len(b.intersection(c)) == 0

        if dataset == 'ADNI_PETdx':
            checkpath1 = os.path.join(BASEDIR, 'utils', 'datacheck', 'ADNI_PETdx' + '_' + opt.clfsetting + '.pkl')
            checkpath2 = os.path.join(BASEDIR, 'utils', 'datacheck', 'ADNI_PET' + '_' + 'regression' + '.pkl')
            with open(checkpath1, 'rb') as f:
                id_infos1 = pkl.load(f)
            with open(checkpath2, 'rb') as f:
                id_infos2 = pkl.load(f)
            a = set(id_infos1[0]).union(set(id_infos2[0]))
            b = set(id_infos1[1]).union(set(id_infos2[1]))
            c = set(id_infos1[2]).union(set(id_infos2[2]))
            assert len(a.intersection(b)) == len(a.intersection(c)) == len(b.intersection(c)) == 0

    return data_train, data_val, data_test
