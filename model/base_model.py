import torch
from einops import rearrange
from typing import Optional, Tuple, Union

from torch import nn
from transformers import CLIPModel as HFCLIPModel, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIP_VISION_INPUTS_DOCSTRING
from transformers.utils import replace_return_docstrings, add_start_docstrings_to_model_forward


# class VT_CLIP(nn.Module):
#     output_dict: torch.jit.Final[bool]
#
#     def __init__(
#             self,
#             embed_dim: int,
#             vision_cfg: CLIPVisionCfg,
#             text_cfg: CLIPTextCfg,
#             quick_gelu: bool = False,
#             cast_dtype: Optional[torch.dtype] = None,
#             output_dict: bool = False,
#     ):
#         super().__init__()
#         self.output_dict = output_dict
#         self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
#
#         text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
#         self.transformer = text.transformer
#         self.context_length = text.context_length
#         self.vocab_size = text.vocab_size
#         self.token_embedding = text.token_embedding
#         self.positional_embedding = text.positional_embedding
#         self.ln_final = text.ln_final
#         self.text_projection = text.text_projection
#         self.register_buffer('attn_mask', text.attn_mask, persistent=False)
#
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#
#
#
#     def unlock_time_attn(self):
#         for name, param in self.named_parameters():
#             if 'time' in name:
#                 param.requires_grad = True
#
#     def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
#         # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
#         self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)
#
#     def lock_text_tower(self, unlocked_layers=0, freeze_layer_norm=False):
#         for param in self.transformer.parameters():
#             param.requires_grad = False
#         for param in self.token_embedding.parameters():
#             param.requires_grad = False
#         for param in self.ln_final.parameters():
#             param.requires_grad = False
#         self.positional_embedding.requires_grad = False
#         self.text_projection.requires_grad = False
#
#         if unlocked_layers != 0:
#             groups = [
#                 [
#                     self.token_embedding,
#                     self.positional_embedding,
#                 ],
#                 *self.transformer.resblocks[:-1],
#                 [
#                     self.transformer.resblocks[-1],
#                     self.ln_final,
#                 ],
#                 self.text_projection,
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
#             _unlock(groups[-unlocked_layers:])
#
#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         self.visual.set_grad_checkpointing(enable)
#         self.transformer.grad_checkpointing = enable
#
#     def encode_image(self, image, normalize: bool = False):
#         features = self.visual(image)
#         return F.normalize(features, dim=-1) if normalize else features
#
#     def encode_text(self, text, normalize: bool = False):
#         cast_dtype = self.transformer.get_cast_dtype()
#
#         x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
#
#         x = x + self.positional_embedding.to(cast_dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x, attn_mask=self.attn_mask)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
#         return F.normalize(x, dim=-1) if normalize else x
#
#     def forward(
#             self,
#             image: Optional[torch.Tensor] = None,
#             text: Optional[torch.Tensor] = None,
#     ):
#         image_features = self.encode_image(image, normalize=True) if image is not None else None
#         text_features = self.encode_text(text, normalize=True) if text is not None else None
#         if self.output_dict:
#             return {
#                 "image_features": image_features,
#                 "text_features": text_features,
#                 "logit_scale": self.logit_scale.exp()
#             }
#         return image_features, text_features, self.logit_scale.exp()
from model.process_clip import get_global_value


class CLIPModel(HFCLIPModel):
    def __init__(self, config, num_frames, add_time_attn):
        super(CLIPModel, self).__init__(config)
        if add_time_attn:
            config.vision_config.num_frames = num_frames
        self.vision_model.forward = self.vision_model_forward

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def vision_model_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.vision_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.vision_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.vision_model.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        if len(pixel_values.shape) == 7:
            b_new, pair_new, T, bs_new, channel_new, h_new, w_new = pixel_values.shape
            # print(pixel_values.shape)
            B = b_new * pair_new * bs_new
            pixel_values = pixel_values.reshape(B*T, channel_new, h_new, w_new)

        elif len(pixel_values.shape) == 5:
            B, _, T, _, _ = pixel_values.shape
            # print(pixel_values.shape)
            pixel_values = rearrange(pixel_values, 'b c t h w -> (b t) c h w')
        else:
            # print(pixel_values.shape)
            B, _, _, _ = pixel_values.shape
            T = 1
        hidden_states = self.vision_model.embeddings(pixel_values)
        #
        # if self.temporal_embedding is not None and get_global_value()['NUM_FRAMES'] != 1:
        #     n = hidden_states.shape[1]
        #     hidden_states = rearrange(hidden_states, '(b t) n d -> (b n) t d', t=T)
        #     hidden_states = hidden_states + self.temporal_embedding[:, :T, :]
        #     hidden_states = rearrange(hidden_states, '(b n) t d -> (b t) n d', n=n)

        hidden_states = self.vision_model.patch_dropout(hidden_states, B, T)
        # print(hidden_states.shape)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)

        pooled_output = pooled_output.reshape(B, T, -1).mean(1)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def encode_image(self, image, normalize: bool = False):
        vision_outputs = self.vision_model(
            pixel_values=image,
            return_dict=True,
        )
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        return image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else image_embeds

    def encode_text(self, input_ids, attention_mask, normalize: bool = False):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        return text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True) if normalize else text_embeds


    def forward(
            self,
            image=None,
            input_ids=None, attention_mask=None
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(input_ids, attention_mask, normalize=True) if input_ids is not None else None
        # if self.output_dict:
        return {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp()
        }
        # return image_features, text_features, self.logit_scale.exp()
