import torch
import torch.nn as nn
import clip
from einops.layers.torch import Rearrange, Reduce


class mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Regressor(nn.Module):
    def __init__(self, num_features, num_classes=1):
        super().__init__()
        self.lnorm = nn.LayerNorm(num_features)
        self.mlp = mlp(num_features, hidden_dim=num_features//2, out_dim=num_features//4, drop=0.5) 
        self.fc = nn.Linear(num_features//4, num_classes)
        self.proj = nn.Linear(num_features, num_features//4)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_p = self.proj(x)
        x = self.mlp(self.lnorm(x))
        x = x + x_p
        pred = self.fc(x)
        return x, self.activation(pred)

class Predictor(nn.Module):
    def __init__(self, num_features, num_classes=2):
        super().__init__()
        self.lnorm = nn.LayerNorm(num_features)
        self.mlp = mlp(num_features, hidden_dim=num_features//2, out_dim=num_features//4, drop=0.5) 
        self.fc = nn.Linear(num_features//4, num_classes)
        self.proj = nn.Linear(num_features, num_features//4)

    def forward(self, x):
        x_p = self.proj(x)
        x = self.mlp(self.lnorm(x))
        x = x + x_p
        pred = self.fc(x)
        return x, pred
       
# Reference: https://github.com/rishikksh20/MLP-Mixer-pytorch/blob/master/mlp-mixer.py
# Reference: https://github.com/lucidrains/mlp-mixer-pytorch/tree/main
class MixFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_feature, expansion = 4, dropout = 0.):
        super().__init__()

        self.feature_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            MixFeedForward(num_feature, num_feature*expansion, dropout),
            Rearrange('b d n -> b n d')
        )
        
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            MixFeedForward(dim, dim*expansion, dropout),
        )
    
    def forward(self, x):
        x = x + self.feature_mix(x)
        x = x + self.channel_mix(x)
        
        return x

class MixerLayer(nn.Module):
    def __init__(self, dim, num_feature, depth= 3,  expansion = 4, dropout = 0.):
        super().__init__()
        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, num_feature, expansion, dropout))
        
        self.layer_norm = nn.LayerNorm(dim)
        self.m_pooling = Reduce('b n c -> b c', 'max')
        
    def forward(self, x):
        
        for mb in self.mixer_blocks:
            x = mb(x)

        x = self.layer_norm(x)
        x = self.m_pooling(x)
        
        return x

class SimpleAgeGenderMixer_farlV1(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        faRL, _ = clip.load("ViT-B/16", device="cpu")
        farl_state=torch.load("/mnt/d/chinghsien/Downloads/NTU_Thesis_Code_backup/FaRL_ckpt/FaRL-Base-Patch16-LAIONFace20M-ep64.pth") # you can download from https://github.com/FacePerceiver/FaRL#pre-trained-backbones
        faRL.load_state_dict(farl_state["state_dict"],strict=False)

        if dataset.split("-")[0] == "utk" :
            self.class_num = 116
        elif dataset.split("-")[0] == "imdb":
            self.class_num = 95
        elif dataset.split("-")[0] == "afad":
            self.class_num = 58
        elif dataset.split("-")[0] == "cacd":
            self.class_num = 49
        elif dataset.split("-")[0] == "agedb":
            self.class_num = 101
        elif dataset.split("-")[0] == "fgnet":
            self.class_num = 70
        elif dataset.split("-")[0] == "clap":
            self.class_num = 96
            
        self.FeatureExtractor = faRL.visual
        self.proj = nn.Linear(512, 128)
        self.gender_classifier = Predictor(512, 2)
        self.age_classifier1 = Regressor(512, 1)
        self.age_classifier2 = nn.Linear(128, self.class_num)
        self.AGmixer = nn.Sequential(
            Rearrange('b (n d) -> b n d', d=128),
            MixerLayer(128, 2, depth=3, dropout=0.5)
        )

    def forward(self, img):
        x = self.FeatureExtractor(img)
        x_norm = x / x.norm(dim=-1, keepdim=True)
        
        gender_feature, gender_out_= self.gender_classifier(x_norm)
        
        age1_input = x_norm 
        age_feature, age_out_1_ = self.age_classifier1(age1_input)
        
        concat_feature  = torch.cat((age_feature, gender_feature), dim=1)
        mix_feature = self.AGmixer(concat_feature)
        
        x_norm = self.proj(x_norm)     
        age2_input = x_norm + mix_feature 
        age_out_2_ = self.age_classifier2(age2_input) 
        
        return age_out_1_, gender_out_, age_out_2_, age_feature, age2_input


def build_model(dataset='utk'):
    
    model = SimpleAgeGenderMixer_farlV1(dataset)   

    return model
