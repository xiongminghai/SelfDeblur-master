import torch
import torch.nn as nn
from torch.autograd import Variable

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# 计算损失函数
class fixed_loss(nn.Module):
    '''
    loss function designed by DIP-net
    '''
    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image):
        h_x = out_image.size()[2]
        w_x = out_image.size()[3]
        count_h = self._tensor_size(out_image[:, :, 1:, :])
        count_w = self._tensor_size(out_image[:, :, :, 1:])
        h_tv = torch.pow((out_image[:, :, 1:, :] - out_image[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((out_image[:, :, :, 1:] - out_image[:, :, :, :w_x - 1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = torch.mean(torch.pow((out_image - gt_image), 2)) +  \
               0.5 * tvloss
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class Matcher:
    def __init__(self, how='gram_matrix', loss='mse', map_index=933):
        self.mode = 'store'
        self.stored = {}
        self.losses = {}

        if how in all_features.keys():
            self.get_statistics = all_features[how]
        else:
            assert False
        pass

        if loss in all_losses.keys():
            self.loss = all_losses[loss]
        else:
            assert False

        self.map_index = map_index
        self.method = 'match'


    def __call__(self, module, features):
        statistics = self.get_statistics(features)

        self.statistics = statistics
        if self.mode == 'store':
            self.stored[module] = statistics.detach()

        elif self.mode == 'match':
           
            if statistics.ndimension() == 2:

                if self.method == 'maximize':
                    self.losses[module] = - statistics[0, self.map_index]
                else:
                    self.losses[module] = torch.abs(300 - statistics[0, self.map_index]) 

            else:
                ws = self.window_size

                t = statistics.detach() * 0

                s_cc = statistics[:1, :, t.shape[2] // 2 - ws:t.shape[2] // 2 + ws, t.shape[3] // 2 - ws:t.shape[3] // 2 + ws] #* 1.0
                t_cc = t[:1, :, t.shape[2] // 2 - ws:t.shape[2] // 2 + ws, t.shape[3] // 2 - ws:t.shape[3] // 2 + ws] #* 1.0
                t_cc[:, self.map_index,...] = 1

                if self.method == 'maximize':
                    self.losses[module] = -(s_cc * t_cc.contiguous()).sum()
                else:
                    self.losses[module] = torch.abs(200 -(s_cc * t_cc.contiguous())).sum()


    def clean(self):
        self.losses = {}

def gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def features(x):
    return x


all_features = {
    'gram_matrix': gram_matrix,
    'features': features,
}

all_losses = {
    'mse': nn.MSELoss(),
    'smoothL1': nn.SmoothL1Loss(),
    'L1': nn.L1Loss(),
}
