import torch
import torch.nn as nn
from models.loss import batch_episym
from models.transformer import TransformerLayer


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    if logits.shape[1] == 2:
        mask = logits[:, 0, :, 0]
        weights = logits[:, 1, :, 0]

        mask = torch.sigmoid(mask)
        weights = torch.exp(weights) * mask
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)
    elif logits.shape[1] == 1:
        weights = torch.relu(torch.tanh(logits))  # tanh and relu

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


class ResNet_Block(nn.Module):
    def __init__(self, in_channels, out_channels, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1)),
            nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (1, 1)),
            nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


class ResNet_Module(nn.Module):
    def __init__(self, channels=128):
        super(ResNet_Module, self).__init__()

        self.embed = nn.Sequential(
            ResNet_Block(channels, channels, pre=False),
            ResNet_Block(channels, channels, pre=False),
            ResNet_Block(channels, channels, pre=False),
            ResNet_Block(channels, channels, pre=False)
        )

    def forward(self, features):
        return self.embed(features)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]

    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx_out + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class ESGA_Module(nn.Module):
    def __init__(self, in_channels=128, k=9):
        super(ESGA_Module, self).__init__()
        self.knn_num = k
        self.reduction = 3

        self.MDB1 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=1, bias=True),  # MLP
            nn.InstanceNorm2d(in_channels, eps=1e-5),  # Dynamic graph Norm
            nn.BatchNorm2d(in_channels)
        )
        self.MDB2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=True),
            nn.InstanceNorm2d(in_channels, eps=1e-5),
            nn.BatchNorm2d(in_channels)
        )
        self.MDB3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=1, stride=1, bias=True),
            nn.InstanceNorm2d(in_channels*2, eps=1e-5),
            nn.BatchNorm2d(in_channels*2)
        )

        self.se = nn.Sequential(
            nn.Conv2d(k, k // self.reduction, kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(k // self.reduction, k, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        dg = get_graph_feature(features, k=self.knn_num)  # generate dynamic graph
        residual = dg

        out = self.relu(self.MDB1(dg))
        dg1 = self.relu(self.MDB2(out))

        att = dg1.mean(dim=2)  # squeeze
        att = self.se(att.unsqueeze(dim=3).transpose(1, 2).contiguous())
        out = torch.mul(out, att.permute(0, 2, 3, 1).contiguous())

        out = self.MDB3(out)
        out += residual

        return self.relu(out)


class AnnularConv(nn.Module):
    def __init__(self, in_channels=128, k=9):
        super(AnnularConv, self).__init__()
        self.in_channel = in_channels
        self.knn_num = k

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        return self.conv(features)


class SSGTransformer(nn.Module):
    def __init__(self, in_channels=128, clusters=96):
        super(SSGTransformer, self).__init__()
        self.conv_group1 = nn.Sequential(
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, clusters, kernel_size=1)
        )
        self.conv_group2 = nn.Sequential(
            nn.InstanceNorm2d(in_channels, eps=1e-3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, clusters, kernel_size=1)
        )
        hidden_dim = clusters
        num_heads = 4
        dropout = None
        activation_fn = 'ReLU'
        self.transformer = TransformerLayer(hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn)

    def forward(self, features):
        embed1 = self.conv_group1(features)
        S1 = torch.softmax(embed1, dim=2).squeeze(3)
        sparse = torch.matmul(features.squeeze(3), S1.transpose(1, 2)).unsqueeze(3)  # graph pool
        residual = sparse

        sparse = self.transformer(sparse.squeeze(dim=3), sparse.squeeze(dim=3))[0].unsqueeze(dim=-1)
        sparse = sparse + residual

        embed2 = self.conv_group2(features)
        S2 = torch.softmax(embed2, dim=1).squeeze(3)
        dense = torch.matmul(sparse.squeeze(3), S2).unsqueeze(3)  # graph unpool

        return dense


class DS_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channels=128, k_num=9, sampling_rate=0.5):
        super(DS_Block, self).__init__()
        self.initial = initial
        self.in_channels = 4 if self.initial is True else 6
        self.out_channels = out_channels
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate
        self.clusters = 96

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels, (1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.res_module1 = nn.Sequential(
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False)
        )
        self.res_module2 = nn.Sequential(
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False),
            ResNet_Block(out_channels, out_channels, pre=False)
        )

        self.sga_module = ESGA_Module(out_channels, k_num)
        self.annular = AnnularConv(out_channels, k_num)

        self.sg_transformer = SSGTransformer(out_channels, self.clusters)

        self.embed = ResNet_Block(out_channels*2, out_channels, pre=True)

        self.linear_0 = nn.Conv2d(out_channels, 1, (1, 1))
        self.linear_1 = nn.Conv2d(out_channels, 1, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):

        B, _, N, _ = x.size()
        indices = indices[:, :int(N*self.sr)]
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)
            w_out = torch.gather(weights, dim=-1, index=indices)
        indices = indices.view(B, 1, -1, 1)

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y, corrs_num, **kwargs):
        B, _, N , _ = x.size()
        out = x.transpose(1, 3).contiguous()
        out = self.conv(out)

        out = self.res_module1(out)
        out = self.sga_module(out)
        out = self.annular(out)
        out = self.res_module2(out)
        feature0 = out
        w0 = self.linear_0(out).view(B, -1)

        out = self.sg_transformer(out)

        out = self.embed(torch.cat([feature0, out], dim=1))
        feature1 = out
        w1 = self.linear_1(out).view(B, -1)

        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N*self.sr)]

            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)

            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds], [feature0, feature1]
        else:
            feature0_0 = kwargs["features0"][0]
            feature0_1 = kwargs["features0"][1]

            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N*self.sr)]
            # feature down_sampling to 1/4
            indices_ds = indices[:, :int(corrs_num * 0.25)]
            indices_ds = indices_ds.view(B, 1, -1, 1)
            feature1_0 = torch.gather(feature0, dim=2, index=indices_ds.repeat(1, 128, 1, 1))
            # feature1_1 = torch.gather(feature1, dim=2, index=indices_ds.repeat(1, 128, 1, 1))
            feature0_0 = torch.gather(feature0_0, dim=2, index=indices_ds.repeat(1, 128, 1, 1))
            feature0_1 = torch.gather(feature0_1, dim=2, index=indices_ds.repeat(1, 128, 1, 1))

            x_ds, y_ds, w0_ds, feature1_1 = self.down_sampling(x, y, w0, indices, feature1, self.predict)

            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds], [feature1_0, feature1_1, feature0_0, feature0_1]


class JPDS_conv(nn.Module):
    def __init__(self, inter_channels):
        super(JPDS_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, inter_channels, (1, 1)),
            nn.BatchNorm2d(inter_channels),
            nn.Conv2d(inter_channels, 1, (1, 1))
        )

    def forward(self, x):
        return self.conv(x)


class SGANet(nn.Module):
    def __init__(self, config):
        super(SGANet, self).__init__()

        self.out_channels = 128

        self.ds_0 = DS_Block(initial=True, predict=False, out_channels=128, k_num=9, sampling_rate=config.sr)
        self.ds_1 = DS_Block(initial=False, predict=True, out_channels=128, k_num=6, sampling_rate=config.sr)

        # graph-context fusion block
        self.conv1 = JPDS_conv(4)
        self.conv2 = JPDS_conv(4)
        self.conv3 = JPDS_conv(4)
        self.conv4 = JPDS_conv(4)
        self.conv5 = JPDS_conv(2)

        self.res = ResNet_Block(self.out_channels, self.out_channels, pre=False)
        self.linear = nn.Conv2d(self.out_channels, 2, (1, 1))

    def forward(self, x, y):
        B, _, N, _ = x.shape

        x1, y1, ws0, w_ds0, features0 = self.ds_0(x, y, N)

        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)

        x2, y2, ws1, w_ds1, features_ds = self.ds_1(x_, y1, N, features0=features0)

        # graph-context fusion block
        features_ds = torch.cat(features_ds, dim=3).transpose(1, 3).contiguous()
        features_ds1 = self.conv1(features_ds)
        features_ds2 = self.conv2(features_ds)
        features_ds3 = self.conv3(features_ds)
        features_ds4 = self.conv4(features_ds)
        features_ds = torch.cat([features_ds1, features_ds2, features_ds3, features_ds4], dim=1)
        features_ds5 = self.conv5(features_ds)

        features_ds = self.res(features_ds5.transpose(1, 3).contiguous())
        w2 = self.linear(features_ds)
        e_hat = weighted_8points(x2, w2)

        with torch.no_grad():
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat)

        return ws0 + ws1 + [w2[:, 0, :, 0]], [y, y, y1, y1, y2], [e_hat], y_hat, [x, x, x1, x1, x2]