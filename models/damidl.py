# based on the released code for the original paper: https://github.com/WyZhuNUAA/DA-MIDL/blob/main/Net/DAMIDL.py
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch
import numpy as np


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.cs = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        a = self.cs(a)
        return x * a


class AttentionBlock(nn.Module):
    def __init__(self, patch_num):
        super(AttentionBlock, self).__init__()
        self.patch_num = patch_num
        self.GlobalAveragePool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.GlobalMaxPool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
        self.Attn = nn.Sequential(
            nn.Conv3d(self.patch_num, self.patch_num // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.patch_num // 2, self.patch_num, kernel_size=1)
        )
        self.pearson_attn = nn.Linear(self.patch_num - 1, 1)

    def forward(self, input, patch_pred):
        mean_input = input.mean(2)
        attn1 = self.Attn(self.GlobalAveragePool(mean_input))
        attn2 = self.Attn(self.GlobalMaxPool(mean_input))
        patch_pred = patch_pred.unsqueeze(-1)
        patch_pred = patch_pred.unsqueeze(-1)
        patch_pred = patch_pred.unsqueeze(-1)
        a = attn1 + attn2 + patch_pred
        a = torch.sigmoid(a)
        return mean_input * a, a.flatten(1)


class BaseNet(nn.Module):
    def __init__(self, feature_depth):
        super(BaseNet, self).__init__()
        self.feature_depth = feature_depth
        self.spatial_attention = SpatialAttention()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, self.feature_depth[0], kernel_size=4)),
            ('norm1', nn.BatchNorm3d(self.feature_depth[0])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(self.feature_depth[0], self.feature_depth[1], kernel_size=3)),
            ('norm2', nn.BatchNorm3d(self.feature_depth[1])),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool3d(kernel_size=2)),
            ('conv3', nn.Conv3d(self.feature_depth[1], self.feature_depth[2], kernel_size=3)),
            ('norm3', nn.BatchNorm3d(self.feature_depth[2])),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv3d(self.feature_depth[2], self.feature_depth[3], kernel_size=3)),
            ('norm4', nn.BatchNorm3d(self.feature_depth[3])),
            ('relu4', nn.ReLU(inplace=True)),
        ]))
        self.classify = nn.Sequential(
            nn.Linear(self.feature_depth[3], 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        local_feature = self.features(input)
        attended_feature = self.spatial_attention(local_feature)
        feature_ = F.adaptive_avg_pool3d(local_feature, (1, 1, 1))
        score = self.classify(feature_.flatten(1, -1))
        return [attended_feature, score]


class DAMIDL(nn.Module):
    def __init__(self, patch_num=50, feature_depth=None):
        super(DAMIDL, self).__init__()
        self.patch_num = patch_num
        self.criterion = nn.CrossEntropyLoss()
        if feature_depth is None:
            feature_depth = [32, 64, 128, 128]
        self.patch_net = BaseNet(feature_depth)
        self.attention_net = AttentionBlock(self.patch_num)
        self.reduce_channels = nn.Sequential(
            nn.Conv3d(self.patch_num, 128, kernel_size=2),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 64, kernel_size=2),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, input):
        patch_feature, patch_score = [], []
        for i in range(self.patch_num):
            feature, score = self.patch_net(input[:, i, :])
            feature = feature.unsqueeze(1)
            patch_feature.append(feature)
            patch_score.append(score)
        feature_maps = torch.cat(patch_feature, 1)
        patch_scores = torch.cat(patch_score, 1)
        attn_feat, ca = self.attention_net(feature_maps, patch_scores)
        features = self.reduce_channels(attn_feat).flatten(1)
        subject_pred = self.fc(features)
        return subject_pred,

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
                group_labels.append(dis_label)

        _probs = torch.cat([j[0] for j in predicts], axis=0).cpu()
        _probs = _probs.squeeze()

        predicts = np.array(
            [np.concatenate([j[i][:, -1:].cpu().numpy() for j in predicts], axis=0)
             for i in range(len(predicts[1]))])

        predicts = predicts.transpose((1, 0, 2))
        groundtruths = np.concatenate(groundtruths, axis=0)
        group_labels = np.concatenate([i.cpu().unsqueeze(-1).numpy() for i in group_labels], axis=0)

        groundtruths = groundtruths[:, :, -1:]
        predicts = predicts[:, :, -1:]

        val_loss = self.criterion(_probs, torch.from_numpy(groundtruths.squeeze().astype(int)))

        return predicts, groundtruths, group_labels, val_loss

    def fit(self, train_loader, optimizer, device, dtype):
        losses = torch.zeros(1, dtype=dtype, device=device, )
        self.train(True)
        for n, data in enumerate(train_loader, 0):
            inputs, aux_labels, labels, dis_label = data

            # multi patch
            labels = labels[:, 0, 0, :]

            inputs = inputs.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=torch.int64)
            optimizer.zero_grad()
            outputs = self(inputs)

            # softmax
            # assert outputs[0].shape == labels.shape
            labels = labels.view(-1)
            non_nan = ~torch.isnan(labels)
            if non_nan.any():
                loss = self.criterion(outputs[0][non_nan], labels[non_nan])
                loss.backward()
                losses[0] += loss.detach()
            optimizer.step()
        return losses / len(train_loader)


if __name__ == '__main__':
    def count_params(model, framework='pytorch'):
        if framework == 'pytorch':
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
        elif framework == 'keras':
            params = model.count_params()
        else:
            raise NotImplementedError
        print('The network has {} params.'.format(params))


    model = DAMIDL(patch_num=50, feature_depth=[32, 64, 128, 128])
    count_params(model)
    img = torch.randn(8, 50, 1, 25, 25, 25)

    pred = model(img)  # (1, 1000)

    subject_pred = model(img)
