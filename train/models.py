

from __future__ import division

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import scipy.sparse as sp
import numpy as np
import os

def get_state(m):
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state


class CT_feather(nn.Module):    # not completed
    def __init__(
    ):
        super(CT_feather, self).__init__()


    def forward(self,CT_image):




class baseline_predict_model(torch.nn.Module):
    def __init__(self,
                CT_embedding_dim = 32,
                other_imformation_dim=64,
                FVC_imformation_dim =64,
                CT_imformation_dim=64,
    ):

        super(baseline_predict_model, self).__init__()

        self.CT_feather_model = CT_feather()

        self.CT_conv = nn.Sequential(
            nn.Conv2d(CT_embedding_dim,CT_imformation_dim,1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fvc_linear = nn.Sequential(
            nn.Linear(146, FVC_imformation_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))

        self.other_imfor_linear = nn.Sequential(
            nn.Linear(3, other_imformation_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5))


        pointwise_in_channels = other_imformation_dim + FVC_imformation_dim + CT_imformation_dim
        self.pointwise = nn.Sequential(
            nn.Conv2d(pointwise_in_channels,32,1),
            nn.Conv2d(32,16,1),
            nn.Conv2d(16,8,1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.output_layer == nn.Sequential(
            nn.Maxpool2d((5,1))
            nn.Linear(146,146)
        )


    def forward(self,CT_image, fvc, other_imfor):
        CT_embedding = self.CT_feather(CT_image)   #32*28*28

        fvc_linear = self.fvc_linear(fvc)
        fVC_reshaped = fvc_linear.view(-1, 64, 1, 1).repeat(1, 1, 28, 28)

        other_linear = self.other_imfor_linear(other_imfor)
        other_reshaped = other_linear.view(-1, 64, 1, 1).repeat(1, 1, 28, 28)

        x_1 = torch.cat((CT_embedding, fVC_reshaped), dim=1)  #64*28*28
        x_2 = torch.cat((x_1, other_reshaped), dim=1)
        x_3 = self.pointwise(x_2)
        output = self.output_layer(x_3)
        return output

