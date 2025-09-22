import pdb
from torch import nn
from torch.nn import functional as F
from .loss_functions import Contrastive_Loss, Cosine_Sim_Loss
import torch
class _DMMI_Framework(nn.Module):
    def __init__(self, backbone, classifier):
        super(_DMMI_Framework, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.cossim = Cosine_Sim_Loss()
        self.contrastive = Contrastive_Loss()

    def forward(self, x, l_feats, l_feats1, l_mask, target_flag=None, training_flag=True):

        input_shape = x.shape[-2:]
        torch.cuda.empty_cache()
        l_1, features = self.backbone(x, l_feats, l_mask)
        torch.cuda.empty_cache()
        x_c1, x_c2, x_c3, x_c4 = features
        de_feat, l_2, x = self.classifier(l_1, l_feats1, x_c4, x_c3, x_c2, x_c1)
        torch.cuda.empty_cache()
        seg_mag = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        torch.cuda.empty_cache()
        if training_flag and target_flag!=None:
            loss_contrastive = self.contrastive(de_feat, l_1, target_flag)
            loss_cossim = self.cossim(l_1, l_2, l_mask, target_flag)
        else:
            loss_contrastive = 0
            loss_cossim = 0
        torch.cuda.empty_cache()
        return loss_contrastive, loss_cossim, seg_mag

class DMMI(_DMMI_Framework):
    pass
