import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Mlp
from timm.layers import DropPath, trunc_normal_


# 线性注意力机制（Performer风格），完善特征映射实现
class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 feature_map='relu'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.feature_map = feature_map

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 特征映射所需的参数
        if feature_map == 'relu':
            self.feat_proj = nn.Linear(head_dim, head_dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # 应用特征映射 φ
        if self.feature_map == 'relu':
            k_prime = F.relu(self.feat_proj(k)) * math.sqrt(self.scale)
            q_prime = F.relu(self.feat_proj(q)) * math.sqrt(self.scale)
        else:  # 简化版特征映射
            k_prime = k
            q_prime = q

        # 线性注意力计算: (q·φ(k))·v
        context = torch.einsum('bhnd,bhne->bhde', k_prime, v)  # [B, num_heads, head_dim, head_dim]
        out = torch.einsum('bhnd,bhde->bhne', q_prime, context)  # [B, num_heads, N, head_dim]

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# 移动窗口局部注意力（Swin风格）
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [M, M]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# 窗口移动函数
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# 混合注意力块
class HybridAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type='window'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.attn_type = attn_type

        assert 0 <= shift_size < window_size, "shift_size必须在[0, window_size)"

        self.norm1 = norm_layer(dim)

        # 根据注意力类型初始化不同的注意力模块
        if attn_type == 'window':
            self.attn = WindowAttention(
                dim, window_size=(window_size, window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        elif attn_type == 'linear':
            self.attn = LinearAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        elif attn_type == 'standard':
            # 标准多头注意力，需要指定embed_dim和num_heads
            self.attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        else:
            raise ValueError(f"不支持的注意力类型: {attn_type}")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.register_buffer("attn_mask", None)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "输入序列长度必须等于H*W"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.attn_type == 'window' and self.shift_size > 0:
            # 移动特征图
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # 创建掩码
            if self.attn_mask is None:
                img_mask = torch.zeros((1, H, W, 1), device=x.device)
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(img_mask, self.window_size)
                mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
                self.register_buffer("attn_mask", attn_mask)
            else:
                attn_mask = self.attn_mask

            # 分割与注意力计算
            x_windows = window_partition(shifted_x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
            attn_windows = self.attn(x_windows, mask=attn_mask)  # 传递x和mask（2个参数）
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            if self.attn_type == 'linear':
                x = x.view(B, H * W, C)
                x = self.attn(x)  # 传递x（1个参数）
                x = x.view(B, H, W, C)
            elif self.attn_type == 'standard':
                # 标准多头注意力需要输入形状为 [batch_size, seq_len, embed_dim]
                x = x.view(B, H * W, C)
                x = self.attn(x, x, x)[0]  # 传递query, key, value（3个参数）
                x = x.view(B, H, W, C)

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# 分层混合注意力Transformer
class HierarchicalHybridTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, has_logits=True):
        super().__init__()
        self.has_logits = has_logits
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size

        # Patch嵌入
        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # 位置编码
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dim))
        self.pos_drop1 = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建各阶段网络
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = self._make_layer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                attn_type='window' if i_layer < 1 else ('linear' if i_layer < 3 else 'standard'),
                downsample=i_layer < self.num_layers - 1
            )
            self.layers.append(layer)

        # 分类头
        self.norm = norm_layer(embed_dim * 2 ** (self.num_layers - 1))
        representation_dim = embed_dim * 2 ** (self.num_layers - 1)

        if has_logits:
            self.pre_logits = nn.Linear(representation_dim, representation_dim)
            self.head = nn.Linear(representation_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(representation_dim, num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        trunc_normal_(self.pos_embed1, std=.02)
        self.apply(self._init_weights)

    def _make_layer(self, dim, depth, num_heads, mlp_ratio, qkv_bias, drop, attn_drop,
                    drop_path, norm_layer, attn_type, downsample=True):
        layers = []
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else self.window_size // 2
            layers.append(HybridAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=self.window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                attn_type=attn_type
            ))
        if downsample:
            layers.append(PatchMerging(dim=dim, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)

        B, L, C = x.shape
        H, W = self.patch_embed1.grid_size

        for i, layer in enumerate(self.layers):
            # 递归处理nn.Sequential中的所有子层
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    if isinstance(sub_layer, (HybridAttentionBlock, PatchMerging)):
                        x = sub_layer(x, H, W)
                    else:
                        x = sub_layer(x)
            else:
                if isinstance(layer, (HybridAttentionBlock, PatchMerging)):
                    x = layer(x, H, W)
                else:
                    x = layer(x)

            if i < self.num_layers - 1:
                H, W = (H + 1) // 2, (W + 1) // 2

        x = self.norm(x)
        if self.has_logits and hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        return x.mean(dim=1)  # [B, C]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# 用于下采样的Patch合并层
class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "输入序列长度必须等于H*W"

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x


# 创建混合注意力模型的函数
def hybrid_attention_transformer(num_classes=1000, has_logits=True, **kwargs):
    """
    创建结合局部注意力和线性注意力的混合Transformer模型
    """
    model = HierarchicalHybridTransformer(
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        num_classes=num_classes,
        has_logits=has_logits,  # 显式传递参数
        **kwargs
    )
    return model