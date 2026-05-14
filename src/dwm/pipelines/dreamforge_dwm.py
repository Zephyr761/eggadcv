"""
DreamForge-DiT Pipeline for simgen3d

This pipeline implements the DreamForge-DiT model with:
1. Motion Frames Conditioning (17 frames: 9 motion + 8 video)
2. LocalMotionAttention for temporal modeling
3. Vector Map support (BEV HDMap and Layout Canvas)

Based on:
- DreamForge-DiT: /DreamForge/WorldDreamer/DreamForge-DiT/
- cross_attention.py: src/dwm/pipelines/cross_attention.py

关键设计原理（来自 DreamForge 文档）:
- 17帧 = 9个motion frames + 8个video frames
- VAE time_downsample_factor = 4，所以17帧 → 5个latent帧
- 前3个latent帧对应motion frames，**不添加噪声**（x_mask机制）
- 后2个latent帧对应video frames，正常扩散训练
"""

import contextlib
import diffusers # Hugging Face Diffusers库，用于扩散模型的训练和推理
import diffusers.image_processor
import dwm.common
import dwm.distributed
import dwm.functional
import dwm.models.crossview_temporal_unet
import dwm.utils.preview
import einops # 用于张量重排的库
import itertools # Python内置库，用于迭代器操作
import math
import os
import re # 用于正则表达式操作
import safetensors.torch # 用于高效加载和保存模型权重的库
import torch.nn.functional as F
import time
import torch
import torch.amp
import torch.distributed.checkpoint.state_dict # 用于分布式训练中的模型权重管理
import torch.distributed.fsdp # 用于分布式训练的库，提供了 Fully Sharded Data Parallel (FSDP) 模式
import torch.distributed.fsdp.sharded_grad_scaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.utils.tensorboard
import torchvision
import transformers
import bisect # Python内置库，用于处理有序列表的插入和查找
from tqdm import tqdm
from einops import rearrange
import torch.nn as nn
from timm.models.vision_transformer import Mlp # 来自 timm 库的 MLP 模块，常用于 Transformer 中的前馈网络

# 从 cross_attention.py 导入 CrossviewTemporalSD 基类
from .cross_attention import CrossviewTemporalSD


# ============================================================================
# LocalMotionAttention 模块 (来自 DreamForge-DiT)
# 参考: dreamforgedit/models/dreamforge/dreamforge_stdit3.py 第45-89行
# ============================================================================

class LocalMotionAttention(nn.Module):
    """
    DreamForge-DiT 中的 LocalMotionAttention 模块

    基于时序差分的运动建模：
    - 计算相邻帧之间的特征差分 (motion)
    - 使用 forward 和 backward 两个方向的差分
    - 通过 MLP 生成注意力权重
    - 应用注意力到原始特征

    参考: DreamForge/WorldDreamer/DreamForge-DiT/dreamforgedit/models/dreamforge/dreamforge_stdit3.py
    """
    def __init__(self, dim, bias=False) -> None:
        super().__init__() # pytorch后台开启自动追踪 自动微分

        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)

        # Forward motion MLP (从当前帧到下一帧的运动)
        self.forward_block = nn.Sequential(
            nn.Linear(dim, dim, bias=bias),
            nn.GELU(),
            nn.Linear(dim, dim, bias=bias),
            nn.Sigmoid()
        )

        # Backward motion MLP (从下一帧到当前帧的运动)
        self.backward_block = nn.Sequential(
            nn.Linear(dim, dim, bias=bias),
            nn.GELU(),
            nn.Linear(dim, dim, bias=bias),
            nn.Sigmoid()
        )

        # Forward 和 backward 方向的可学习权重
        # 注意：原始代码直接初始化为 torch.ones(2)/2
        self.learnable_param = nn.Parameter(torch.ones(2) / 2)

    def forward(self, x):
        """
        Args:
            x: [B, T, C] batch, time, channels

        Returns:
            outputs: [B, T, C] motion-modulated features
        """
        B, T, C = x.shape

        # Query-Key-Value 投影

        #q表示当前帧需要关注的特征 k表示相邻帧需要给当前帧的上下文特征 v是原始特征

        hidden_states_in = self.to_qkv(x)
        hs_q, hs_k, hs_v = torch.chunk(hidden_states_in, 3, dim=-1)
        # 与传统注意力机制不同 这里用差分代替点乘计算注意力权重
        # Forward motion: t_{i+1} - t_i
        motion_forward = torch.cat([
            torch.zeros_like(hs_q[:, :1]),  # 第一帧没有前向motion(没有前一帧)
            hs_q[:, 1:] - hs_k[:, :-1]  # 后一帧的query与当前帧的key计算差分
        ], dim=1)  # [B, T, C]

        # 计算 forward attention
        attn_forward = self.forward_block(motion_forward.flatten(0, 1)) # 底层处理2D数据的速度更快
        attn_forward = attn_forward.view(B, T, C)

        # Backward motion: t_{i-1} - t_i
        motion_backward = torch.cat([
            hs_q[:, :-1] - hs_k[:, 1:],  # 最后一帧没有backward motion(没有下一帧)
            torch.zeros_like(hs_q[:, -1:]) # 当前帧的query与后一帧的key计算差分
        ], dim=1)  # [B, T, C]

        # 计算 backward attention
        attn_backward = self.backward_block(motion_backward.flatten(0, 1))
        attn_backward = attn_backward.view(B, T, C)

        # 合并 forward 和 backward attention（可学习权重）
        attn = self.learnable_param[0] * attn_forward + \
               self.learnable_param[1] * attn_backward  # [B, T, C]

        # 应用 attention 到 value
        outputs = attn * hs_v

        return outputs


# ============================================================================
# x_mask 生成函数 (用于 DreamForge 训练)
# 参考: train_dreamforge_t.py 第591-608行
# ============================================================================

def compute_motion_frame_mask(T, ref_length=9, video_length=8, vae_temporal_stride=4, device='cpu'):
    """
    生成 motion frames 的 mask

    DreamForge 使用 VAE 的时间压缩 (time_downsample_factor = 4)：
    - 输入：T帧 (17 = ref_length + video_length)
    - 输出：ceil(T / vae_temporal_stride) 个 latent 帧
    - 前 motion_latent_frames = ceil(ref_length / vae_temporal_stride) 帧设为 False（不添加噪声）
    - 后面帧设为 True（正常扩散训练）

    Args:
        T: 输入帧数（默认17）
        ref_length: motion frames 数量（默认9）
        video_length: video frames 数量（默认8）
        vae_temporal_stride: VAE时间下采样因子（默认4）
        device: 设备

    Returns:
        mask: [1, latent_frames] bool tensor，False表示motion frames区域
    """
    # VAE时间压缩：17帧 → ceil(17/4) = 5个latent帧
    total_latent_frames = (T + vae_temporal_stride - 1) // vae_temporal_stride

    # motion frames对应的latent数量: ceil(9/4) = 3
    motion_latent_frames = (ref_length + vae_temporal_stride - 1) // vae_temporal_stride

    # 创建mask [1, latent_frames]
    mask = torch.full((1, total_latent_frames), True, dtype=torch.bool, device=device)

    # 前3个latent对应motion frames，设为False（不添加噪声）
    mask[0, :motion_latent_frames] = False

    return mask


# ============================================================================
# DreamForge-DWM Pipeline 主类
# ============================================================================

class DreamForgeDWM(CrossviewTemporalSD):
    """
    DreamForge-DWM Pipeline

    基于 CrossviewTemporalSD，添加 DreamForge 特性：
    1. Motion Frames (17帧: 9 motion + 8 video)
    2. x_mask 机制：motion frames 不添加噪声
    3. LocalMotionAttention for temporal modeling
    4. Vector Map support (hdmap_image_settings)
    5. Layout Canvas (projected map + boxes)

    关键设计：
    - Motion Frames 不单独编码，而是和 video frames 一起通过VAE
    - 使用 x_mask 区分 motion frames 和 video frames
    - Motion frames 区域在训练时不添加噪声（x_mask=False）
    - 支持端到端训练（所有模型参数可训练，除非有 freezing_pattern 配置）
    """

    # DreamForge 特有参数(kwargs)列表（不传给父类）
    DREAMFORGE_KWARGS = {
        'use_motion_frames', 'ref_length', 'video_length', 'candidate_length',
        'vae_temporal_stride', 'use_local_motion_attention'
    }

    def __init__(self, *args, **kwargs):
        # 提取 DreamForge 特有的参数
        self.use_motion_frames = kwargs.pop('use_motion_frames', True)
        self.ref_length = kwargs.pop('ref_length', 9)
        self.video_length = kwargs.pop('video_length', 8)
        self.candidate_length = kwargs.pop('candidate_length', 8) 
        self.use_local_motion_attention = kwargs.pop('use_local_motion_attention', True)

        # 父类初始化（不使用 DreamForge 特有参数）
        super().__init__(*args, **kwargs) #无名称的参数作为元组传给arg按位置解包 关键字参数作为字典传给kwargs按名称解包

        # 总帧数 = ref_length + video_length
        self.total_frames = self.ref_length + self.video_length  # 17

        # VAE 编码后，原始帧数通过时间下采样因子压缩
        # total_frames -> latent_frames = ceil(total_frames / vae_temporal_stride)
        # 17 -> ceil(17/4) = 5 latent frames
        # 其中前 motion_latent_frames = ceil(ref_length / vae_temporal_stride) = ceil(9/4) = 3 是 motion frames
        # 后面的是 video frames = 5 - 3 = 2
        self.total_latent_frames = (self.total_frames + self.vae_temporal_stride - 1) // self.vae_temporal_stride
        self.motion_latent_frames = (self.ref_length + self.vae_temporal_stride - 1) // self.vae_temporal_stride  # 3
        self.video_latent_frames = self.total_latent_frames - self.motion_latent_frames  # 2

        # DreamForge 需要训练整个模型，重新启用所有参数的可训练状态
        # 父类会冻结所有参数，只解冻 ref/bev 相关的模块
        # 对于 DreamForge，我们解冻所有模型参数进行端到端训练
        self._setup_trainable_parameters()

    def _setup_trainable_parameters(self):
        """设置模型参数的可训练状态 - 只启用 MTA 模块学习，其他全部冻结"""
        # 检查是否有 freezing_pattern 配置，如果有则按配置处理
        training_config = self.common_config if hasattr(self, 'common_config') else {}
        freezing_pattern = training_config.get("freezing_pattern")

        if freezing_pattern:
            # 如果有 custom freezing_pattern(json配置文件)，按配置执行
            import re
            pattern = re.compile(freezing_pattern)
            frozen_module_count = 0
            for name, module in self.model.named_modules():
                if pattern.match(name) is not None:
                    module.requires_grad_(False)
                    frozen_module_count += 1
                    if self.should_save:
                        print("{} is frozen.".format(name))
                else:
                    module.requires_grad_(True)

            if self.should_save:
                print("{} modules are frozen.".format(frozen_module_count))
        else:
            # 默认行为：冻结所有模型参数，只解冻 local_motion_attention (MTA)
            # 1. 先冻结所有参数
            for p in self.model.parameters():
                p.requires_grad = False

            # 2. 只解冻 local_motion_attention 模块（MTA）
            if hasattr(self.model, 'local_motion_attention') and self.model.local_motion_attention is not None:
                trainable_params_count = 0
                for lma in self.model.local_motion_attention:
                    for p in lma.parameters():
                        p.requires_grad = True
                        trainable_params_count += p.numel() # 统计可训练参数数量

                if self.should_save:
                    print("LocalMotionAttention (MTA) module is trainable: {:.1f} M parameters".format(trainable_params_count / 1e6))
            else:
                if self.should_save:
                    print("Warning: local_motion_attention not found in model, all parameters remain frozen.")

            # 3. 如果父类之前已经解冻了 ref/bev 模块，重新冻结它们
            # 重新冻结 ref_transformer_blocks
            if hasattr(self.model, "ref_transformer_blocks"):
                for blk in self.model.ref_transformer_blocks:
                    for p in blk.parameters():
                        p.requires_grad = False

            # 重新冻结 ref_proj 相关
            if hasattr(self.model, "ref_proj_rear"):
                self.model.ref_proj_rear.requires_grad_(False)
            if hasattr(self.model, "ref_proj_front"):
                self.model.ref_proj_front.requires_grad_(False)

            # 重新冻结 ref_mat_pe 相关
            if hasattr(self.model, "ref_mat_pe_front"):
                self.model.ref_mat_pe_front.requires_grad_(False)
            if hasattr(self.model, "ref_mat_pe_rear"):
                self.model.ref_mat_pe_rear.requires_grad_(False)

            # 重新冻结 bev_proj
            if hasattr(self.model, "bev_proj"):
                self.model.bev_proj.requires_grad_(False)

            # 重新冻结 ref_mixers
            if hasattr(self.model, "ref_mixers"):
                for mx in self.model.ref_mixers:
                    for p in mx.parameters():
                        p.requires_grad = False

            # 重新冻结 refcoef
            if hasattr(self.model, "refcoef"):
                self.model.refcoef.requires_grad_(False)

            if self.should_save:
                print("All model parameters are frozen except LocalMotionAttention (MTA).")

        # 打印可训练参数数量
        if self.should_save:
            param_count = sum([
                i.numel() for i in self.model.parameters() if i.requires_grad
            ])
            print("{:.1f} M parameters are trainable.".format(param_count / 1e6))

    def _setup_motion_attention(self):
        """为现有模型添加 LocalMotionAttention 支持"""
        # 如果模型是 DiT 类型，添加 motion attention
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                if hasattr(block, 'temporal_attn'):
                    # 添加 LocalMotionAttention
                    if not hasattr(block, 'local_motion_attn'):
                        dim = block.hidden_size if hasattr(block, 'hidden_size') else \
                             block.temporal_attn.embed_dim
                        block.local_motion_attn = LocalMotionAttention(
                            dim, bias=False
                        )
                        block.motion_proj = nn.Linear(dim, dim)

    def _prepare_layout_canvas(self, batch):
        """
        准备 layout canvas (投影的 vector map + 3D boxes)

        Args:
            batch: data batch

        Returns:
            layout_latents: [B, T, V, C_l, H, W] layout condition latents
        """
        # 这个函数把向量图和3D框投影到BEV空间，得到一个类似于图像的布局画布（layout canvas），然后通过控制网络或embedder编码成潜在表示，作为条件输入到扩散模型中。
        if 'layout_canvas' not in batch:
            return None

        # Layout canvas 已经是投影好的 map + box
        # 需要通过控制网络/embedder 编码
        layout = batch['layout_canvas']  # [B, T, V, H, W, 13] 已经是bev视图

        if hasattr(self, 'controlnet'):
            B, T, V, H, W, C = layout.shape # V表示视角数 C是语义通道数
            layout = rearrange(layout, "B T V H W C -> (B T V) C H W")

            # 如果有 layout embedder
            if hasattr(self.controlnet, 'layout_embedder'):# 把layout编码到隐空间
                with torch.no_grad(): # layout_embedder一般是训练好的
                    layout_latents = self.controlnet.layout_embedder(layout)
                    layout_latents = rearrange(
                        layout_latents,
                        "(B T V) C H W -> B T V C H W",
                        B=B, T=T, V=V
                    )
                    return layout_latents

        return None

    def get_conditions(
        self,
        model,
        text_encoder,
        tokenizer,
        common_config: dict,
        latent_shape,
        batch: dict,
        device,
        dtype,
        text_condition_mask=None,
        _3dbox_condition_mask=None,
        hdmap_condition_mask=None,
        action_condition_mask=None,
        explicit_view_modeling_mask=None,
        streaming_mode: bool = False,
        prev_ego_transforms=None,
        do_classifier_free_guidance: bool = False,
        latents_shape=None,
    ):
        """
        获取所有条件，包括 DreamForge 特有的 layout

        注意：Motion frames 不作为单独的条件，而是通过 x_mask 机制处理
        """
        # 调用父类获取基础条件
        batch = batch.copy()

        # 准备 layout canvas
        layout_latents = self._prepare_layout_canvas(batch)
        if layout_latents is not None:
            batch['layout_latents'] = layout_latents

        # 调用父类方法获取其他条件
        conditions = self.get_conditions.__func__(
            model, text_encoder, tokenizer, common_config,
            latent_shape, batch, device, dtype, text_condition_mask,
            _3dbox_condition_mask, hdmap_condition_mask, action_condition_mask,
            explicit_view_modeling_mask, streaming_mode, prev_ego_transforms,
            do_classifier_free_guidance, latents_shape,
        )

        # 添加 layout canvas
        if layout_latents is not None:
            conditions['layout_canvas'] = layout_latents

        return conditions

    def train_step(self, batch: dict, global_step: int):
        """
        重写父类 train_step，添加 DreamForge x_mask 机制
        """
        # 准备 batch 复制
        batch = batch.copy()

        # 计算输入帧数 T (应该是 17 = ref_length + video_length)
        B, T, NC = batch["vae_images"].shape[:3]

        # 生成 x_mask: 前3个latent帧对应motion frames，不添加噪声
        if self.use_motion_frames:
            device = batch["vae_images"].device
            x_mask = compute_motion_frame_mask(
                T, self.ref_length, self.video_length, self.vae_temporal_stride, device
            )
            # 将 x_mask 添加到 batch
            batch['x_mask'] = x_mask

        # 调用父类的原始 train_step 方法
        return super().train_step(batch, global_step)

    def inference_step(self, latents, timestep, batch, **kwargs):
        """
        推理步骤，包含 x_mask 机制

        推理时：使用前一clip生成的最后9帧作为 motion frames
        参考: run_fastapi_dit_t.py 第879-890行
        """
        # 准备 batch 复制
        batch = batch.copy()

        # 推理时也需要 x_mask
        if self.use_motion_frames:
            # 获取当前帧数
            T = latents.shape[2] if latents.dim() >= 3 else latents.shape[1]
            device = latents.device

            x_mask = compute_motion_frame_mask(
                T, self.ref_length, self.video_length, self.vae_temporal_stride, device
            )
            batch['x_mask'] = x_mask

        # 调用父类推理步骤
        return super().inference_step(latents, timestep, batch, **kwargs)


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'DreamForgeDWM',
    'LocalMotionAttention',
    'compute_motion_frame_mask',
]
