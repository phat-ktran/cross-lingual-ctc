import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from .common import Activation

def trunc_normal_(tensor, mean=0., std=1.):
    return init.trunc_normal_(tensor, mean=mean, std=std, a=-2*std, b=2*std)

def normal_(tensor, mean=0., std=1.):
    return init.normal_(tensor, mean=mean, std=std)

def zeros_(tensor):
    return init.zeros_(tensor)

def ones_(tensor):
    return init.ones_(tensor)

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()
        nn.init.kaiming_uniform_(self.conv.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Module):
    def forward(self, x):
        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvMixer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        HW=[8, 25],
        local_k=[3, 3],
    ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        padding = [local_k[0] // 2, local_k[1] // 2]
        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            kernel_size=local_k,
            stride=1,
            padding=padding,
            groups=num_heads,
        )
        nn.init.kaiming_normal_(self.local_mixer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        h, w = self.HW
        x = x.transpose(1, 2).reshape(-1, self.dim, h, w)
        x = self.local_mixer(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mixer="Global",
        HW=None,
        local_k=[7, 11],
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mixer = mixer
        if HW is not None:
            H, W = HW
            self.H = H
            self.W = W
            self.N = H * W
            if mixer == "Local":
                hk, wk = local_k
                mask = torch.ones(H * W, H + hk - 1, W + wk - 1)
                for h in range(H):
                    for w in range(W):
                        mask[h * W + w, h : h + hk, w : w + wk] = 0
                mask = mask[:, hk // 2 : H + hk // 2, wk // 2 : W + wk // 2].reshape(H * W, H * W)
                mask[mask == 1] = float('-inf')
                self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.mixer == "Local":
            attn += self.mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mixer="Global",
        local_mixer=[7, 11],
        HW=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer="GELU",
        norm_layer="LayerNorm",
        epsilon=1e-6,
        prenorm=True,
    ):
        super().__init__()
        norm_layer_class = getattr(torch.nn, norm_layer)
        self.norm1 = norm_layer_class(dim, eps=epsilon)
        if mixer in ["Global", "Local"]:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif mixer == "Conv":
            self.mixer = ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("mixer must be one of ['Global', 'Local', 'Conv']")
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer_class(dim, eps=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=getattr(torch.nn, act_layer),
            drop=drop,
        )
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=[32, 100],
        in_channels=3,
        embed_dim=768,
        sub_num=2,
        patch_size=[4, 4],
        mode="pope",
    ):
        super().__init__()
        if mode == "pope":
            if sub_num == 2:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels,
                        embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                        act=nn.GELU,
                    ),
                    ConvBNLayer(
                        embed_dim // 2,
                        embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                        act=nn.GELU,
                    ),
                )
            elif sub_num == 3:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels,
                        embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                        act=nn.GELU,
                    ),
                    ConvBNLayer(
                        embed_dim // 4,
                        embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                        act=nn.GELU,
                    ),
                    ConvBNLayer(
                        embed_dim // 2,
                        embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                        act=nn.GELU,
                    ),
                )
            else:
                raise ValueError("sub_num must be 2 or 3 for pope mode")
            downsample_factor = 2 ** sub_num
            self.num_patches = (img_size[0] // downsample_factor) * (img_size[1] // downsample_factor)
        elif mode == "linear":
            self.proj = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
            self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        else:
            raise ValueError("mode must be 'pope' or 'linear'")
        self.img_size = img_size
        self.embed_dim = embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SubSample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        types="Pool",
        stride=[2, 1],
        sub_norm="LayerNorm",
        act=None,
    ):
        super().__init__()
        self.types = types
        if types == "Pool":
            self.avgpool = nn.AvgPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2]
            )
            self.maxpool = nn.MaxPool2d(
                kernel_size=[3, 5], stride=stride, padding=[1, 2]
            )
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            )
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        norm_layer = getattr(nn, sub_norm)
        self.norm = norm_layer(out_channels)
        self.act = act() if act is not None else None

    def forward(self, x):
        if self.types == "Pool":
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            x = x.flatten(2).transpose(1, 2)
            out = self.proj(x)
        else:
            x = self.conv(x)
            out = x.flatten(2).transpose(1, 2)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out

class Im2Seq(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.permute((0, 2, 1))  # (NTC)(batch, width, channels)
        return x


class CTCHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=6625,
        fc_decay=0.0004,
        mid_channels=None,
        return_feats=False,
        **kwargs,
    ):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                bias=True,
            )
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                bias=True,
            )
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                bias=True,
            )

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, labels=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts

        if not self.training:
            predicts = torch.nn.functional.softmax(predicts, dim=2)
            result = predicts

        return result


class SVTRNet(nn.Module):
    def __init__(
        self,
        img_size=[32, 100],
        in_channels=3,
        embed_dim=[64, 128, 256],
        depth=[3, 6, 3],
        num_heads=[2, 4, 8],
        mixer=["Local"] * 6 + ["Global"] * 6,  # Local atten, Global atten, Conv
        local_mixer=[[7, 11], [7, 11], [7, 11]],
        patch_merging="Conv",  # Conv, Pool, None
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        last_drop=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer="LayerNorm",
        sub_norm="LayerNorm",
        epsilon=1e-6,
        out_channels=192,
        out_char_num=25,
        block_unit="Block",
        act="GELU",
        last_stage=True,
        sub_num=2,
        prenorm=True,
        use_lenhead=False,
        vocab_size=6625,
        fc_decay=0.0004,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.vocab_size = vocab_size
        self.prenorm = prenorm
        patch_merging = (
            None
            if patch_merging != "Conv" and patch_merging != "Pool"
            else patch_merging
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num,
        )
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2**sub_num), img_size[1] // (2**sub_num)]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)
        Block_unit = eval(block_unit)

        dpr = np.linspace(0, drop_path_rate, sum(depth))
        self.blocks1 = nn.ModuleList(
            [
                Block_unit(
                    dim=embed_dim[0],
                    num_heads=num_heads[0],
                    mixer=mixer[0 : depth[0]][i],
                    HW=self.HW,
                    local_mixer=local_mixer[0],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=act,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[0 : depth[0]][i],
                    norm_layer=norm_layer,
                    epsilon=epsilon,
                    prenorm=prenorm,
                )
                for i in range(depth[0])
            ]
        )
        if patch_merging is not None:
            self.sub_sample1 = SubSample(
                embed_dim[0],
                embed_dim[1],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging,
            )
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.ModuleList(
            [
                Block_unit(
                    dim=embed_dim[1],
                    num_heads=num_heads[1],
                    mixer=mixer[depth[0] : depth[0] + depth[1]][i],
                    HW=HW,
                    local_mixer=local_mixer[1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=act,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depth[0] : depth[0] + depth[1]][i],
                    norm_layer=norm_layer,
                    epsilon=epsilon,
                    prenorm=prenorm,
                )
                for i in range(depth[1])
            ]
        )
        if patch_merging is not None:
            self.sub_sample2 = SubSample(
                embed_dim[1],
                embed_dim[2],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging,
            )
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        self.blocks3 = nn.ModuleList(
            [
                Block_unit(
                    dim=embed_dim[2],
                    num_heads=num_heads[2],
                    mixer=mixer[depth[0] + depth[1] :][i],
                    HW=HW,
                    local_mixer=local_mixer[2],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=act,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depth[0] + depth[1] :][i],
                    norm_layer=norm_layer,
                    epsilon=epsilon,
                    prenorm=prenorm,
                )
                for i in range(depth[2])
            ]
        )
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d([1, out_char_num])
            self.last_conv = nn.Conv2d(
                in_channels=embed_dim[2],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.hardswish = Activation("hard_swish", inplace=True)  # nn.Hardswish()
            # self.dropout = nn.Dropout(p=last_drop, mode="downscale_in_infer")
            self.dropout = nn.Dropout(p=last_drop)
        if not prenorm:
            self.norm = eval(norm_layer)(embed_dim[-1], eps=epsilon)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = Activation(
                "hard_swish", inplace=True
            )  # nn.Hardswish()
            self.dropout_len = nn.Dropout(p=last_drop)

        self.neck = Im2Seq(self.out_channels)

        self.fc_deay = fc_decay
        self.head = CTCHead(self.neck.out_channels, self.vocab_size, self.fc_deay)

        torch.nn.init.xavier_normal_(self.pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # weight initialization
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(
                x.permute(0, 2, 1).reshape(
                    [-1, self.embed_dim[0], self.HW[0], self.HW[1]]
                )
            )
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(
                x.permute(0, 2, 1).reshape(
                    [-1, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]
                )
            )
        for blk in self.blocks3:
            x = blk(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))

        if self.last_stage:
            if self.patch_merging is not None:
                h = self.HW[0] // 4
            else:
                h = self.HW[0]
            x = self.avg_pool(
                x.permute(0, 2, 1).reshape([-1, self.embed_dim[2], h, self.HW[1]])
            )
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)

        x = self.neck(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    model = SVTRNet()
