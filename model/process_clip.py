import logging
import math
from typing import Optional, Tuple
from einops import rearrange
from peft import LoraConfig, get_peft_model
from transformers import CLIPConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer as SpatialCLIPEncoderLayer, CLIPAttention, CLIPMLP
import torch
from torch import nn
from torch.nn import functional as F

from training.distributed import is_master

aaa = {'NUM_FRAMES': 1, 'PATCH_DROPOUT': 0.0}

def set_global_value(k, v):
    global aaa
    aaa[k] = v

def get_global_value():
    global aaa
    return aaa

# @dataclass
# class CLIPVisionCfg:
#     layers: Union[Tuple[int, int, int, int], int] = 12
#     width: int = 768
#     head_width: int = 64
#     mlp_ratio: float = 4.0
#     patch_size: int = 16
#     image_size: Union[Tuple[int, int], int] = 224
#     cast_dtype: str = None
#     num_frames: int = 2
#
#     ls_init_value: Optional[float] = None  # layer scale initial value
#     patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
#     input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
#     global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
#     attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
#     n_queries: int = 256  # n_queries for attentional pooler
#     attn_pooler_heads: int = 8  # n heads for attentional_pooling
#     output_tokens: bool = False
#
#     timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
#     timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
#     timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
#     timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
#     timm_proj_bias: bool = False  # enable bias final projection
#     timm_drop: float = 0.  # head dropout
#     timm_drop_path: Optional[float] = None  # backbone stochastic depth

# class Video_VisionTransformer(nn.Module):
#     output_tokens: torch.jit.Final[bool]
#
#     def __init__(
#             self,
#             num_frames: int,
#             image_size: int,
#             patch_size: int,
#             width: int,
#             layers: int,
#             heads: int,
#             mlp_ratio: float,
#             ls_init_value: float = None,
#             global_average_pool: bool = False,
#             attentional_pool: bool = False,
#             n_queries: int = 256,
#             attn_pooler_heads: int = 8,
#             output_dim: int = 512,
#             patch_dropout: float = 0.,
#             input_patchnorm: bool = False,
#             act_layer: Callable = nn.GELU,
#             norm_layer: Callable = LayerNorm,
#             output_tokens: bool = False
#     ):
#         super().__init__()
#         self.output_tokens = output_tokens
#         image_height, image_width = self.image_size = to_2tuple(image_size)
#         patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
#         self.grid_size = (image_height // patch_height, image_width // patch_width)
#         self.output_dim = output_dim
#
#         # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
#         self.input_patchnorm = input_patchnorm
#
#         if input_patchnorm:
#             patch_input_dim = patch_height * patch_width * 3
#             self.patchnorm_pre_ln = LayerNorm(patch_input_dim)
#             self.conv1 = nn.Linear(patch_input_dim, width)
#         else:
#             self.patchnorm_pre_ln = nn.Identity()
#             self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size,
#                                    bias=False)
#
#         # class embeddings and positional embeddings
#         self.scale = scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
#
#         self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
#         # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
#         self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
#
#         self.ln_pre = norm_layer(width)
#         self.transformer = Transformer(
#             width,
#             layers,
#             heads,
#             mlp_ratio,
#             ls_init_value=ls_init_value,
#             act_layer=act_layer,
#             norm_layer=norm_layer,
#         )
#
#         self.global_average_pool = global_average_pool
#         if attentional_pool:
#             self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries)
#             self.ln_post = norm_layer(output_dim)
#             self.proj = nn.Parameter(scale * torch.randn(output_dim, output_dim))
#         else:
#             self.attn_pool = None
#             self.ln_post = norm_layer(width)
#             self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
#
#         self.init_parameters()
#
#
#     def lock(self, unlocked_groups=0, freeze_bn_stats=False):
#         for param in self.parameters():
#             param.requires_grad = False
#
#         if unlocked_groups != 0:
#             groups = [
#                 [
#                     self.conv1,
#                     self.positional_embedding,
#                     self.ln_pre,
#                 ],
#                 *zip(self.transformer.resblocks[:-1], [self.class_embedding for i in range(len(self.transformer.resblocks[:-1]))]),
#                 [
#                     self.class_embedding,
#                     self.transformer.resblocks[-1],
#                     self.ln_post,
#                 ],
#                 [self.proj, self.temporal_embedding]
#             ]
#
#             def _unlock(x):
#                 if isinstance(x, Sequence):
#                     for g in x:
#                         _unlock(g)
#                 else:
#                     if isinstance(x, torch.nn.Parameter):
#                         x.requires_grad = True
#                     else:
#                         for p in x.parameters():
#                             p.requires_grad = True
#
#             _unlock(groups[-unlocked_groups:])
#
#     def init_parameters(self):
#         # FIXME OpenAI CLIP did not define an init for the VisualTransformer
#         # TODO experiment if default PyTorch init, below, or alternate init is best.
#
#         nn.init.normal_(self.temporal_embedding, std=self.scale)
#         # nn.init.normal_(self.class_embedding, std=self.scale)
#         # nn.init.normal_(self.positional_embedding, std=self.scale)
#         #
#         # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
#         # attn_std = self.transformer.width ** -0.5
#         # fc_std = (2 * self.transformer.width) ** -0.5
#         # for block in self.transformer.resblocks:
#         #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
#         #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
#         #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#         #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
#         #
#         # if self.text_projection is not None:
#         #     nn.init.normal_(self.text_projection, std=self.scale)
#         # pass
#
#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         self.transformer.grad_checkpointing = enable
#
#     def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         if self.global_average_pool:
#             return x.mean(dim=1), x
#         else:
#             return x[:, 0], x[:, 1:]
#
#     def forward(self, x: torch.Tensor):
#         # print('input img', x.shape)
#         B, _, T, _, _ = x.shape
#         x = rearrange(x, 'b c t h w -> (b t) c h w')
#         # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
#         if self.input_patchnorm:
#             # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
#             x = x.reshape(x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0], self.grid_size[1],
#                           self.patch_size[1])
#             x = x.permute(0, 2, 4, 1, 3, 5)
#             x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
#             x = self.patchnorm_pre_ln(x)
#             x = self.conv1(x)
#         else:
#             x = self.conv1(x)  # shape = [*, width, grid, grid]
#             x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#             x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#
#         # print('embed img', x.shape)
#         # class embeddings and positional embeddings
#         x = torch.cat(
#             [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
#              x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.positional_embedding.to(x.dtype)
#
#         n = x.shape[1]
#         x = rearrange(x, '(b t) n d -> (b n) t d', t=T)
#         x = x + self.temporal_embedding[:, :T, :]
#         x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
#
#         # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
#         x = self.patch_dropout(x)
#         x = self.ln_pre(x)
#
#         # print('patch_dropout img', x.shape)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         # print('permute img', x.shape)
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#
#         if self.attn_pool is not None:
#             x = self.attn_pool(x)
#             x = self.ln_post(x)
#             pooled, tokens = self._global_pool(x)
#         else:
#             pooled, tokens = self._global_pool(x)
#             pooled = self.ln_post(pooled)  # bt, d
#
#         pooled = pooled.reshape(B, T, -1).mean(1)
#         if self.proj is not None:
#             pooled = pooled @ self.proj
#
#         if self.output_tokens:
#             return pooled, tokens
#
#         return pooled
#
# def _build_vision_tower(
#         embed_dim: int,
#         vision_cfg: CLIPVisionCfg,
#         quick_gelu: bool = False,
#         cast_dtype: Optional[torch.dtype] = None
# ):
#     if isinstance(vision_cfg, dict):
#         vision_cfg = CLIPVisionCfg(**vision_cfg)
#
#     # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
#     # memory efficient in recent PyTorch releases (>= 1.10).
#     # NOTE: timm models always use native GELU regardless of quick_gelu flag.
#     act_layer = QuickGELU if quick_gelu else nn.GELU
#
#     vision_heads = vision_cfg.width // vision_cfg.head_width
#     norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
#     visual = Video_VisionTransformer(
#         num_frames=vision_cfg.num_frames,
#         image_size=vision_cfg.image_size,
#         patch_size=vision_cfg.patch_size,
#         width=vision_cfg.width,
#         layers=vision_cfg.layers,
#         heads=vision_heads,
#         mlp_ratio=vision_cfg.mlp_ratio,
#         ls_init_value=vision_cfg.ls_init_value,
#         patch_dropout=vision_cfg.patch_dropout,
#         input_patchnorm=vision_cfg.input_patchnorm,
#         global_average_pool=vision_cfg.global_average_pool,
#         attentional_pool=vision_cfg.attentional_pool,
#         n_queries=vision_cfg.n_queries,
#         attn_pooler_heads=vision_cfg.attn_pooler_heads,
#         output_tokens=vision_cfg.output_tokens,
#         output_dim=embed_dim,
#         act_layer=act_layer,
#         norm_layer=norm_layer,
#     )
#
#     return visual




class CLIPEncoderLayer(SpatialCLIPEncoderLayer):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, config.num_frames, config.hidden_size))
        nn.init.normal_(self.temporal_embedding, std=config.hidden_size ** -0.5)

        self.embed_dim = config.hidden_size
        self.temporal_attn = CLIPAttention(config)
        # self.temporal_mlp = CLIPMLP(config)
        # self.t_attn_gate = nn.Parameter(torch.tensor([-20.]))
        # self.t_ffn_gate = nn.Parameter(torch.tensor([-20.]))
        self.temporal_layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # self.temporal_layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """


        bt, n, d = hidden_states.shape
        t = get_global_value()['NUM_FRAMES']


        # time embed
        if t != 1:
            n = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=t)
            hidden_states = hidden_states + self.temporal_embedding[:, :t, :]
            hidden_states = rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)

        # time attn
        residual = hidden_states
        hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=t)
        # hidden_states = self.layer_norm1(hidden_states)  # share layernorm
        hidden_states = self.temporal_layer_norm1(hidden_states)
        hidden_states, attn_weights = self.temporal_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)

        # residual = hidden_states
        # hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=t)
        # # hidden_states = self.layer_norm2(hidden_states)  # share layernorm
        # hidden_states = self.temporal_layer_norm2(hidden_states)
        # hidden_states = self.temporal_mlp(hidden_states)
        # hidden_states = residual + rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)

        # spatial attn
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs





# class ResidualAttentionBlock(SpatialResidualAttentionBlock):
#     def __init__(self,
#                  num_frames: int,
#                  d_model: int,
#                  n_head: int,
#                  mlp_ratio: float = 4.0,
#                  ls_init_value: float = None,
#                  act_layer: Callable = nn.GELU,
#                  norm_layer: Callable = LayerNorm,
#                  is_cross_attention: bool = False,):
#         super().__init__(d_model, n_head, mlp_ratio, ls_init_value, act_layer, norm_layer, is_cross_attention)
#
#         self.num_frames = num_frames
#         self.time_ln_1 = norm_layer(d_model)
#         self.time_attn = nn.MultiheadAttention(d_model, n_head)
#         self.time_ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
#
#     def time_attention(
#             self,
#             q_x: torch.Tensor,
#             k_x: Optional[torch.Tensor] = None,
#             v_x: Optional[torch.Tensor] = None,
#             attn_mask: Optional[torch.Tensor] = None,
#     ):
#         k_x = k_x if k_x is not None else q_x
#         v_x = v_x if v_x is not None else q_x
#
#         attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
#         return self.time_attn(
#             q_x, k_x, v_x, need_weights=True, attn_mask=attn_mask
#         )[0]
#
#     def forward(
#             self,
#             q_x: torch.Tensor,
#             k_x: Optional[torch.Tensor] = None,
#             v_x: Optional[torch.Tensor] = None,
#             attn_mask: Optional[torch.Tensor] = None,
#     ):
#         k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
#         v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
#
#         n, bt, d = q_x.shape
#         t = get_global_value()['NUM_FRAMES']
#
#         # time attn
#         # print('q_x', q_x.shape)
#         xt = rearrange(q_x, 'n (b t) d -> t (b n) d', t=t)
#         # print('xt', xt.shape)
#         xt = self.time_ls_1(self.time_attention(q_x=self.time_ln_1(xt), k_x=None, v_x=None, attn_mask=None))
#         # print('time_attention xt', xt.shape)
#         q_x = q_x + rearrange(xt, 't (b n) d -> n (b t) d', n=n)
#         # print('time_attention q_x', xt.shape)
#
#         # spatial attn
#         x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
#
#         x = x + self.ls_2(self.mlp(self.ln_2(x)))
#         return x

def print_trainable_parameters(model, msg=''):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(f"{msg} Trainable params: {trainable_params} || all params: {all_param} || "
                 f"trainable: {100 * trainable_params / all_param:.2f}%")

def convert_model_to_lora(args, model):
    if args.clip_type == 'vl' and args.add_time_attn:
        target_modules = ["temporal_attn.k_proj", "temporal_attn.v_proj",
                          "temporal_attn.q_proj", "temporal_attn.out_proj",
                          "temporal_mlp.fc1", "temporal_mlp.fc2"]
    else:
        target_modules = ["k_proj", "v_proj", "q_proj", "out_proj"]
    config = LoraConfig(
        r=args.lora_r,         # 16
        lora_alpha=args.lora_alpha,  #  16
        target_modules=target_modules,  # self_attn.out_proj
        lora_dropout=args.lora_dropout,        # 0.1
        bias="none",
        modules_to_save=[],
    )
    model.vision_model.encoder.is_gradient_checkpointing = False
    model.vision_model.encoder = get_peft_model(model.vision_model.encoder, config)
    if is_master(args):
        print_trainable_parameters(model.vision_model.encoder, msg='The model.vision_model.encoder: ')
    # model.text_model.encoder.is_gradient_checkpointing = False
    # model.text_model.encoder = get_peft_model(model.text_model.encoder, config)
    # if is_master(args):
    #     print_trainable_parameters(model.text_model.encoder, msg='The model.text_model.encoder: ')



def add_time_attn_block(m: nn.ModuleList, device):
    config = m.config
    for i, sub_m in enumerate(m.layers):
        if isinstance(sub_m, SpatialCLIPEncoderLayer):
            oup = CLIPEncoderLayer(config).to(device)
            state_dict = sub_m.state_dict()

            new_state_dict = {}
            for k, v in state_dict.items():
                if 'self_attn' in k:
                    new_state_dict[k] = v
                    # if 'out_proj' in k:
                    #     v = torch.zeros_like(v, dtype=v.dtype, device=v.device)
                    new_k = 'temporal_attn.' + '.'.join(k.split('.')[1:])
                    new_state_dict[new_k] = v
                elif 'mlp' in k:
                    new_state_dict[k] = v
                    # if 'out_proj' in k:
                    #     v = torch.zeros_like(v, dtype=v.dtype, device=v.device)
                    new_k = 'temporal_mlp.' + '.'.join(k.split('.')[1:])
                    new_state_dict[new_k] = v
                elif 'layer_norm1' in k:
                    new_state_dict[k] = v
                    new_k = 'temporal_layer_norm1.' + '.'.join(k.split('.')[1:])
                    new_state_dict[new_k] = v
                elif 'layer_norm2' in k:
                    new_state_dict[k] = v
                    new_k = 'temporal_layer_norm2.' + '.'.join(k.split('.')[1:])
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v

            missing_keys, unexpected_keys = oup.load_state_dict(new_state_dict, strict=False)
            # assert missing_keys == ["t_attn_gate", "t_ffn_gate"]
            assert missing_keys == ['temporal_embedding']
            assert unexpected_keys == []
            m.layers[i] = oup

def resize_pos(m: nn.Module, args):
    # convert embedding
    if args.clip_type == 'al':
        m.image_size = [args.num_mel_bins, args.target_length]
    m.config.image_size = [m.image_size, m.image_size] if isinstance(m.image_size, int) else m.image_size

    # m.config.num_channels = 1
    # new_patch_embedding = nn.Conv2d(
    #                         in_channels=m.config.num_channels,
    #                         out_channels=m.embed_dim,
    #                         kernel_size=m.patch_size,
    #                         stride=m.patch_size,
    #                         bias=False,
    #                     )
    # state_dict = m.patch_embedding.state_dict()
    # for k, v in state_dict.items():
    #     state_dict[k] = torch.mean(v, dim=1, keepdim=True).to(v.dtype)
    # m.patch_embedding = new_patch_embedding
    # m.patch_embedding.load_state_dict(state_dict)

    # pos resize
    old_pos_embed_state_dict = m.position_embedding.state_dict()
    old_pos_embed = old_pos_embed_state_dict['weight']
    dtype = old_pos_embed.dtype
    grid_size = [m.config.image_size[0] // m.patch_size, m.config.image_size[1] // m.patch_size]
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        m.to(args.device)
        return

    m.num_patches = grid_size[0] * grid_size[1]
    m.num_positions = m.num_patches + 1
    m.register_buffer("position_ids", torch.arange(m.num_positions).expand((1, -1)))
    new_position_embedding = nn.Embedding(m.num_positions, m.embed_dim)

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = [int(math.sqrt(len(pos_emb_img)))]*2

    if is_master(args):
        logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode='bicubic',
        antialias=True,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    old_pos_embed_state_dict['weight'] = new_pos_embed.to(dtype)
    m.position_embedding = new_position_embedding
    m.position_embedding.load_state_dict(old_pos_embed_state_dict)

    m.to(args.device)


# def i2v_linear_resize_pos_embed(state_dict, model, interpolation: str = 'linear', antialias: bool = True):
#     # Rescale the grid of position embeddings when loading from state_dict
#     old_pos_embed = state_dict.get('visual.positional_embedding', None)
#     if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
#         return
#     # grid_size = to_2tuple(model.visual.grid_size)
#     grid_size = model.visual.grid_size
#     extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
#     # new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
#     new_seq_len = grid_size[0] * grid_size[1] * grid_size[2] + extra_tokens
#     if new_seq_len == old_pos_embed.shape[0]:
#         return
#
#     if extra_tokens:
#         pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
#     else:
#         pos_emb_tok, pos_emb_img = None, old_pos_embed
#     # old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))
#
#     logging.info('Resizing position embedding grid-size from %s to %s', old_pos_embed.shape[0], new_seq_len)
#     # pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
#     pos_emb_img = pos_emb_img.unsqueeze(0).permute(0, 2, 1)
#     pos_emb_img = F.interpolate(
#         pos_emb_img,
#         # size=grid_size,
#         size=new_seq_len - extra_tokens,
#         mode=interpolation,
#         # antialias=antialias,
#         # align_corners=False,
#     )
#     # pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
#     pos_emb_img = pos_emb_img.permute(0, 2, 1)[0]
#     if pos_emb_tok is not None:
#         new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
#     else:
#         new_pos_embed = pos_emb_img
#     state_dict['visual.positional_embedding'] = new_pos_embed
#
# def inflate_patch_embed(state_dict, model):
#     old_patch_embed_shape = model.visual.conv1.weight.shape
#     new_patch_embed_shape = state_dict['visual.conv1.weight'].shape
#     if old_patch_embed_shape == new_patch_embed_shape:
#         return
#     expanded_weight = state_dict['visual.conv1.weight'].unsqueeze(2).repeat(1, 1, 2, 1, 1)
#     state_dict['visual.conv1.weight'] = expanded_weight
#
#
# def load_checkpoint(model, pretrained, strict=True):
#     state_dict = load_state_dict(pretrained)
#     # detect old format and make compatible with new format
#     if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
#         state_dict = convert_to_custom_text_state_dict(state_dict)
#     i2v_linear_resize_pos_embed(state_dict, model)
#     inflate_patch_embed(state_dict, model)
#     incompatible_keys = model.load_state_dict(state_dict, strict=strict)
#     return incompatible_keys

