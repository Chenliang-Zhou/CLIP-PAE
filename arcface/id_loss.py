import torch
from torch import nn

from .model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print("Loading ResNet ArcFace ...", end=" ")
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load("../pretrained/model_ir_se50.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        print("Done.")

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y).detach()  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        return torch.stack([1 - y_hat_feats[i].dot(y_feats[i]) for i in range(n_samples)])
