from utils import ops
from torch import nn 
import torch.nn.functional as F
import torch
import modelReACGAN as m
import types
import numpy as np

def model_builder():
    imsize = 64
    def define_modules(gan_config):
        layers = types.SimpleNamespace()
        if gan_config.apply_g_sn:
            layers.g_conv2d = ops.snconv2d
            layers.g_deconv2d = ops.sndeconv2d
            layers.g_linear = ops.snlinear
            layers.g_embedding = ops.sn_embedding
        else:
            layers.g_conv2d = ops.conv2d
            layers.g_deconv2d = ops.deconv2d
            layers.g_linear = ops.linear
            layers.g_embedding = ops.embedding
    
        if gan_config.apply_d_sn:
            layers.d_conv2d = ops.snconv2d
            layers.d_deconv2d = ops.sndeconv2d
            layers.d_linear = ops.snlinear
            layers.d_embedding = ops.sn_embedding
        else:
            layers.d_conv2d = ops.conv2d
            layers.d_deconv2d = ops.deconv2d
            layers.d_linear = ops.linear
            layers.d_embedding = ops.embedding
    
        if gan_config.g_cond_mtd == "cBN":
            layers.g_bn = ops.ConditionalBatchNorm2d
        elif gan_config.g_cond_mtd == "W/O":
            layers.g_bn = ops.batchnorm_2d
        else:
            raise NotImplementedError
    
        if not gan_config.apply_d_sn:
            layers.d_bn = ops.batchnorm_2d
    
        if gan_config.g_act_fn == "ReLU":
            layers.g_act_fn = nn.ReLU(inplace=True)
        elif gan_config.g_act_fn == "Leaky_ReLU":
            layers.g_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif gan_config.g_act_fn == "ELU":
            layers.g_act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif gan_config.g_act_fn == "GELU":
            layers.g_act_fn = nn.GELU()
        elif gan_config.g_act_fn == "Auto":
            pass
        else:
            raise NotImplementedError
    
        if gan_config.d_act_fn == "ReLU":
            layers.d_act_fn = nn.ReLU(inplace=True)
        elif gan_config.d_act_fn == "Leaky_ReLU":
            layers.d_act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif gan_config.d_act_fn == "ELU":
            layers.d_act_fn = nn.ELU(alpha=1.0, inplace=True)
        elif gan_config.d_act_fn == "GELU":
            layers.d_act_fn = nn.GELU()
        elif gan_config.g_act_fn == "Auto":
            pass
        else:
            raise NotImplementedError
        return layers
        
    config = types.SimpleNamespace()
    config.d_act_fn = "ReLU"
    config.g_act_fn = "ReLU"
    config.apply_d_sn= True
    config.apply_g_sn= True
    config.g_cond_mtd= "cBN"
    
    l=define_modules(config)
    
    discriminator = m.Discriminator(imsize,128,True,True,[1],"D2DCE",2048,True,num_classes=1050,d_init="ortho",d_depth=2,mixed_precision=True,MODULES=l)
    
    generator = m.Generator(100,128,imsize,128,True,[4],"cBN",num_classes=1050,g_init="ortho",g_depth=2,mixed_precision=True,MODULES=l)
    return (discriminator,generator)



class Data2DataCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature, m_p, device):
        super(Data2DataCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.m_p = m_p
        self.device = device
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def make_index_matrix(self, labels):
        labels = labels.detach().cpu().numpy()
        num_samples = labels.shape[0]
        mask_multi, target = np.ones([self.num_classes, num_samples]), 0.0

        for c in range(self.num_classes):
            c_indices = np.where(labels==c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.device)

    def remove_diag(self, M):
        h, w = M.shape
        assert h==w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.device)
        return M[mask].view(h, -1)

    def forward(self, embed, proxy, label, **_):
        # calculate similarities between sample embeddings
        sim_matrix = self.calculate_similarity_matrix(embed, embed) + self.m_p - 1
        # remove diagonal terms
        sim_matrix = self.remove_diag(sim_matrix/self.temperature)
        # for numerical stability
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = F.relu(sim_matrix) - sim_max.detach()

        # calculate similarities between sample embeddings and the corresponding proxies
        smp2proxy = self.cosine_similarity(embed, proxy)
        # make false negative removal
        removal_fn = self.remove_diag(self.make_index_matrix(label)[label])
        # apply the negative removal to the similarity matrix
        improved_sim_matrix = removal_fn*torch.exp(sim_matrix)

        # compute positive attraction term
        pos_attr = F.relu((self.m_p - smp2proxy)/self.temperature)
        # compute negative repulsion term
        neg_repul = torch.log(torch.exp(-pos_attr) + improved_sim_matrix.sum(dim=1))
        # compute data to data cross-entropy criterion
        criterion = pos_attr + neg_repul
        return criterion.mean()
def d_hinge(d_logit_real, d_logit_fake):
    return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))
def g_hinge(d_logit_fake):
    return -torch.mean(d_logit_fake)