# based on https://github.com/lucidrains/mlp-mixer-pytorch
# by Zi-Chao Zhang
from torch import nn
from functools import partial
import torch
import numpy as np
import math
import torch.nn.functional as F
import os


class GELU(nn.Module):
    """
    https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/gelu.py
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


class MLPMixer3D(nn.Module):
    def __init__(self, num_patches, channels, patch_size, dim, depth, num_classes,
                 expansion_factor=4, dropout=0.):
        super(MLPMixer3D, self).__init__()

        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.criterion = nn.BCELoss()
        self.embed = nn.Linear(int(np.prod(patch_size) * channels), dim)

        self.mixer = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
            ) for _ in range(depth)],
        )

        self.ln0 = nn.LayerNorm(dim)
        self.output = nn.Sequential(nn.Linear(dim, num_classes), nn.Sigmoid())

    def embeding(self, x):
        # x: batch, num_patch, channel, patch_size, patch_size, patch_size

        x = x.view(*x.shape[:2], -1)
        x = self.embed(x)

        x = self.mixer(x)

        x = self.ln0(x)
        x = x.mean(axis=1)
        return x,

    def forward(self, x):
        # x: batch, num_patch, channel, patch_size, patch_size, patch_size

        x = x.view(*x.shape[:2], -1)
        x = self.embed(x)

        x = self.mixer(x)

        x = self.ln0(x)
        x = x.mean(axis=1)
        return self.output(x),

    def evaluate_data(self, val_loader, device, dtype='float32'):
        predicts = []
        groundtruths = []
        group_labels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data
                inputs = inputs.to(device=device, dtype=dtype)
                outputs = self(inputs)
                predicts.append(outputs)
                groundtruths.append(labels.numpy()[:, 0, :])  # multi patch
                group_labels.append(dis_label[:, 0])

        _probs = torch.stack([torch.cat([j[i] for j in predicts], dim=0) for i in range(len(predicts[1]))], dim=0)
        _probs = _probs.transpose(0, 1).cpu()

        predicts = np.array(
            [np.concatenate([j[i].cpu().numpy() for j in predicts], axis=0) for i in range(len(predicts[1]))])
        predicts = predicts.transpose((1, 0, 2))
        groundtruths = np.concatenate(groundtruths, axis=0)
        group_labels = np.concatenate([i.cpu().unsqueeze(-1).numpy() for i in group_labels], axis=0)

        for i, standr in enumerate(val_loader.dataset.standrs):
            predicts[:, i, :] = standr.unstandr(predicts[:, i, :])
            groundtruths[:, i, :] = standr.unstandr(groundtruths[:, i, :])

        groundtruths = groundtruths[:, :, -1:]
        predicts = predicts[:, :, -1:]

        non_nan = [torch.from_numpy(~np.isnan(groundtruths[:, i, :])) for i in range(groundtruths.shape[1])]
        val_loss = sum([self.criterion(_probs[:, i, :][non_nan[i]], torch.from_numpy(groundtruths[:, i, :])[non_nan[i]])
                        for i in range(groundtruths.shape[1])])

        return predicts, groundtruths, group_labels, val_loss

    def embed_data(self, val_loader, device, dtype='float32'):
        predicts = []
        clflabels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data
                inputs = inputs.to(device=device, dtype=dtype)
                outputs = self.embeding(inputs)
                predicts.append(outputs)
                clflabels.append(labels[:, 0, -1].view(-1))  # multi patch)

        fea = torch.cat([torch.cat([j[i] for j in predicts], dim=0) for i in range(len(predicts[1]))], dim=1)
        fea = fea.cpu().numpy()
        clflabels = torch.cat(clflabels, dim=-1).cpu().numpy()
        return fea, clflabels

    def fit(self, train_loader, optimizer, device, dtype):
        losses = torch.zeros(train_loader.dataset.labels.shape[1], dtype=dtype, device=device, )
        self.train(True)
        for n, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_label = data

            # multi patch
            labels = labels[:, 0, :]

            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=dtype)
            optimizer.zero_grad()
            outputs = self(inputs)

            for i in range(labels.shape[1]):
                assert outputs[i].shape == labels[:, i, :].shape
                non_nan = ~torch.isnan(labels[:, i, :])
                if non_nan.any():
                    loss = self.criterion(outputs[i][non_nan], labels[:, i, :][non_nan])
                    loss.backward(retain_graph=True)
                    losses[i] += loss.detach()
            optimizer.step()
        return losses / len(train_loader)


class RegMLPMixer3D(MLPMixer3D):
    def __init__(self, num_patches, channels, patch_size, dim, depth, num_classes,
                 expansion_factor=4, dropout=0., num_regmixer=1):
        super(RegMLPMixer3D, self).__init__(num_patches, channels, patch_size, dim, depth - num_regmixer, num_classes,
                                            expansion_factor, dropout)
        self.criterion = nn.MSELoss()
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        self.fea1 = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
            ) for _ in range(num_regmixer)]
        )
        self.fea2 = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
            ) for _ in range(num_regmixer)]
        )

        self.ln1 = nn.LayerNorm(dim)
        self.out1 = nn.Sequential(nn.Linear(dim, num_classes),
                                  nn.Sigmoid())

        self.ln2 = nn.LayerNorm(dim)
        self.out2 = nn.Sequential(nn.Linear(dim, num_classes),
                                  nn.Sigmoid())

    def embeding(self, x):
        # x: batch, num_patch, channel, patch_size, patch_size, patch_size

        x = x.view(*x.shape[:2], -1)
        x = self.embed(x)
        x = self.mixer(x)
        fea1 = self.fea1(x)
        fea2 = self.fea2(x)

        fea1 = self.ln1(fea1)
        fea2 = self.ln2(fea2)
        fea1 = fea1.mean(axis=1)
        fea2 = fea2.mean(axis=1)
        return fea1, fea2

    def forward(self, x):
        # x: batch, num_patch, channel, patch_size, patch_size, patch_size

        x = x.view(*x.shape[:2], -1)
        x = self.embed(x)
        x = self.mixer(x)
        fea1 = self.fea1(x)
        fea2 = self.fea2(x)

        fea1 = self.ln1(fea1)
        fea2 = self.ln2(fea2)
        fea1 = fea1.mean(axis=1)
        fea2 = fea2.mean(axis=1)

        return self.out1(fea1), self.out2(fea2)


class FuseMlp(nn.Module):
    def __init__(self, num_patches, channels, patch_size, dim, depth, num_classes,
                 expansion_factor=4, dropout=0., pool=1, clfpat_index=None):
        super(FuseMlp, self).__init__()
        self.criterion = nn.BCELoss()
        if clfpat_index is not None:
            self.clfpat_index = torch.from_numpy(clfpat_index)
        else:
            self.clfpat_index = torch.arange(num_patches)
        self.reg_net = RegMLPMixer3D(num_patches, channels, patch_size, dim, depth, num_classes,
                                     expansion_factor, dropout)
        self.clf_net = MLPMixer3D(len(self.clfpat_index), channels, patch_size, dim, depth, num_classes,
                                  expansion_factor, dropout)

        self.dim = dim
        self.pool = pool

        self.num_classes = num_classes
        fc_size = dim * 3
        self.drop = nn.Dropout(p=0.5)

        self.clfmlp = nn.Sequential(nn.Linear(fc_size // pool, num_classes))

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def embed_data(self, val_loader, device, dtype='float32'):
        predicts = []
        clflabels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data
                inputs = inputs.to(device=device, dtype=dtype)
                outputs = self.embeding(inputs)
                predicts.append(outputs)
                clflabels.append(labels[:, 0, -1].view(-1))  # multi patch)

        fea = torch.cat([torch.cat([j[i] for j in predicts], dim=0) for i in range(len(predicts[1]))], dim=1)
        fea = fea.cpu().numpy()
        clflabels = torch.cat(clflabels, dim=-1).cpu().numpy()
        return fea, clflabels

    def embeding(self, x):
        x = x.view(*x.shape[:2], -1)
        if not self.clfpat_index.device == x.device:
            self.clfpat_index.to(x.device)
        x1 = self.clf_net.embed(x[:, self.clfpat_index, :])
        x1 = self.clf_net.mixer(x1)
        x1 = self.clf_net.ln0(x1)
        clf_fea = x1.mean(axis=1)

        # reg features
        x2 = self.reg_net.embed(x)
        x2 = self.reg_net.mixer(x2)
        fea1 = self.reg_net.fea1(x2)
        fea2 = self.reg_net.fea2(x2)

        fea1 = self.reg_net.ln1(fea1)
        fea2 = self.reg_net.ln2(fea2)
        fea1 = fea1.mean(axis=1)
        fea2 = fea2.mean(axis=1)

        reg_out = [fea1, fea2]

        if self.num_classes == 1:
            clf_fea = torch.cat(reg_out + [clf_fea], dim=-1)
            if self.pool > 1:
                clf_fea = F.avg_pool1d(clf_fea.unsqueeze(-2), kernel_size=self.pool, stride=self.pool).squeeze(-2)
        else:
            raise NotImplementedError
        return clf_fea,

    def forward(self, x):
        x = x.view(*x.shape[:2], -1)
        if not self.clfpat_index.device == x.device:
            self.clfpat_index.to(x.device)
        x1 = self.clf_net.embed(x[:, self.clfpat_index, :])
        x1 = self.clf_net.mixer(x1)
        x1 = self.clf_net.ln0(x1)
        clf_fea = x1.mean(axis=1)

        x2 = self.reg_net.embed(x)
        x2 = self.reg_net.mixer(x2)
        fea1 = self.reg_net.fea1(x2)
        fea2 = self.reg_net.fea2(x2)

        fea1 = self.reg_net.ln1(fea1)
        fea2 = self.reg_net.ln2(fea2)
        fea1 = fea1.mean(axis=1)
        fea2 = fea2.mean(axis=1)

        reg_out = [fea1, fea2]

        if self.num_classes == 1:
            clf_fea = torch.cat(reg_out + [clf_fea], dim=-1)
            if self.pool > 1:
                clf_fea = F.avg_pool1d(clf_fea.unsqueeze(-2), kernel_size=self.pool, stride=self.pool).squeeze(-2)

            clf_fea = self.drop(clf_fea)
            clf = F.sigmoid(self.clfmlp(clf_fea))
        else:
            raise NotImplementedError

        return [clf]

    def evaluate_data(self, val_loader, device, dtype='float32'):
        predicts = []
        groundtruths = []
        group_labels = []

        with torch.no_grad():
            self.train(False)
            for i, data in enumerate(val_loader, 0):
                inputs, aux_labels, labels, dis_label = data
                inputs = inputs.to(device=device, dtype=dtype)
                outputs = self(inputs)
                predicts.append(outputs)
                groundtruths.append(labels.numpy()[:, 0, :])  # multi patch
                group_labels.append(dis_label[:, 0])

        _probs = torch.cat([j[0] for j in predicts], axis=0).cpu()
        _probs = _probs.squeeze()

        predicts = np.array(
            [np.concatenate([j[i].cpu().numpy() for j in predicts], axis=0) for i in range(len(predicts[1]))])
        predicts = predicts.transpose((1, 0, 2))
        groundtruths = np.concatenate(groundtruths, axis=0)
        group_labels = np.concatenate([i.cpu().unsqueeze(-1).numpy() for i in group_labels], axis=0)

        for i, standr in enumerate(val_loader.dataset.standrs):
            predicts[:, i, :] = standr.unstandr(predicts[:, i, :])
            groundtruths[:, i, :] = standr.unstandr(groundtruths[:, i, :])

        groundtruths = groundtruths[:, :, -1:]
        predicts = predicts[:, :, -1:]
        val_loss = self.criterion(_probs, torch.from_numpy(groundtruths.squeeze()))

        return predicts, groundtruths, group_labels, val_loss

    def fit(self, train_loader, optimizer, device, dtype):
        losses = torch.zeros(train_loader.dataset.labels.shape[1], dtype=dtype, device=device, )
        self.train(True)
        for n, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_label = data

            # multi patch
            labels = labels[:, 0, :]

            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=dtype)
            optimizer.zero_grad()
            outputs = self(inputs)

            for i in range(labels.shape[1]):
                assert outputs[i].shape == labels[:, i, :].shape
                non_nan = ~torch.isnan(labels[:, i, :])
                if non_nan.any():
                    loss = self.criterion(outputs[i][non_nan], labels[:, i, :][non_nan])
                    loss.backward(retain_graph=True)
                    losses[i] += loss.detach()
            optimizer.step()
        return losses / len(train_loader)


def fusemlp(**kwargs):
    regpre = os.environ.get('REGPREPATH')
    clfpre = os.environ.get('CLFPREPATH')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FuseMlp(channels=1, num_classes=1, **kwargs)

    model.reg_net.load_state_dict(
        torch.load(regpre, map_location=device), strict=False)
    print('reg pretrain loaded')

    model.clf_net.load_state_dict(
        torch.load(clfpre, map_location=device), strict=False)
    print('clf pretrain loaded')

    for param in model.reg_net.children():
        param.requires_grad_(False)
    return model


def clfmlpm(**kwargs):
    return MLPMixer3D(channels=1, num_classes=1, **kwargs)


def regmlpm(**kwargs):
    return RegMLPMixer3D(channels=1, num_classes=1, **kwargs)
