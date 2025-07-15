# Copied and modified from diffusers/blob/main/src/diffusers/models/attention_processor.py

import copy
import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import gc
from typing import Optional
from einops import rearrange
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

from tqdm import tqdm


class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # new args
        timestep_ind: Optional[int] = None,
        save_attn_weight: bool = False,
        save_attn_kv: bool = False,
        update_attn_kv: bool = False,
        save_attn_out_for_sim: bool = False,
        save_attn_out_for_interpolate: bool = False,
        save_cross_sim: bool = False,
        fg_inter_img_attn: bool = False,
        bg_inter_img_attn: bool = False,
        attn_out_interpolate: bool = False,
        interpolate_weight_dict: dict = dict(),
        spatial_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class CharaConsistAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        size=(64, 64),
        text_seq_len=512,
        # Attention Share
        fg_share_flag=False,
        bg_share_flag=False,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.text_seq_len = text_seq_len
        self.visual_seq_len = size[0] * size[1]
        # Hyper Params
        self.fg_share_flag = fg_share_flag
        self.bg_share_flag = bg_share_flag
        # Sample Info
        self.bg_len = 0
        self.real_len = 0
        self.size = size
        # Save id info
        self.id_attn_bank = dict()
        self.attn_weights = dict()
        self.cross_sims = None
            
    def get_curr_attn_weights(self, q, k):
        attn_weights = torch.matmul(q[:, :, self.text_seq_len:], k[:, :, :self.real_len].transpose(-2, -1)).to(torch.float16)
        attn_weights = attn_weights.softmax(-1).sum(1)
        attn_weights = rearrange(attn_weights, "b (h w) s -> b s h w", h=self.size[0], w=self.size[1])
        bg_attn_weights = attn_weights[:, :self.bg_len].mean(1)
        fg_attn_weights = attn_weights[:, self.bg_len:].mean(1)       
        return bg_attn_weights, fg_attn_weights
    
    def get_curr_cross_sim(self, curr_hidden_states, timestep_ind):
        id_hidden_states = self.id_attn_bank[timestep_ind]["attn_out"].to(curr_hidden_states.device, non_blocking=True)
        sim = torch.matmul(
            F.normalize(curr_hidden_states, dim=-1),
            F.normalize(id_hidden_states, dim=-1).transpose(-2, -1))
        return sim
    
    def get_expand_attn_mask(self, id_fg_mask, curr_fg_mask, bg_share_flag, fg_share_flag, device):
        id_len = len(id_fg_mask)
        _id_fg_mask = id_fg_mask.view(1, 1, 1, -1)
        _curr_fg_mask = curr_fg_mask.view(1, 1, -1, 1)
        expand_mask = _id_fg_mask != _curr_fg_mask
        if not bg_share_flag:
            expand_mask[:, :, :, ~id_fg_mask] = True
        elif not fg_share_flag:
            expand_mask[:, :, :, id_fg_mask] = True
        t2i_expand_mask = torch.ones((1, 1, self.text_seq_len, id_len), device=device, dtype=bool)
        return torch.cat((t2i_expand_mask, expand_mask), dim=-2)
    
    def get_expanded_key_value(
            self,
            image_rotary_emb,
            bg_share_flag,
            fg_share_flag,
            timestep_ind,
            device, 
            id_fg_mask=None,
            id_bg_mask=None,
            curr_fg_mask=None,
            id_fg_inds=None,
            id_bg_inds=None,
            curr_fg_inds=None,
            **kwargs):
        
        saved_key, saved_value = None, None
        id_fg_mask = id_fg_mask.flatten().to(device, non_blocking=True)
        curr_fg_mask = curr_fg_mask.flatten().to(device, non_blocking=True)

        ori_saved_key = self.id_attn_bank[timestep_ind]["key"].to(device, non_blocking=True)
        ori_saved_value = self.id_attn_bank[timestep_ind]["value"].to(device, non_blocking=True)

        if bg_share_flag:
            assert id_bg_mask is not None
            id_bg_mask = id_bg_mask.flatten().to(device, non_blocking=True)
            shared_bg = (id_bg_mask & (~curr_fg_mask))
            if "updated_key" in self.id_attn_bank[timestep_ind]:
                saved_key = self.id_attn_bank[timestep_ind]["updated_key"].to(device, non_blocking=True)[:, :, shared_bg]
                saved_value = self.id_attn_bank[timestep_ind]["updated_value"].to(device, non_blocking=True)[:, :, shared_bg]
            else:
                saved_key = ori_saved_key[:, :, shared_bg]
                saved_value = ori_saved_value[:, :, shared_bg]
            image_rotary_emb_bg = (
                image_rotary_emb[0][shared_bg],
                image_rotary_emb[1][shared_bg],
            )
            saved_key = apply_rotary_emb(saved_key, image_rotary_emb_bg)
            id_fg_mask = torch.zeros([saved_key.shape[2]], device=device, dtype=torch.bool)
        
        if fg_share_flag:
            saved_key_fg = ori_saved_key[:, :, id_fg_inds]
            saved_value_fg = ori_saved_value[:, :, id_fg_inds]
            image_rotary_emb_fg = (
                image_rotary_emb[0][curr_fg_inds],
                image_rotary_emb[1][curr_fg_inds],
            )
            saved_key_fg = apply_rotary_emb(saved_key_fg, image_rotary_emb_fg)
            if saved_key is not None:
                saved_key = torch.cat([saved_key, saved_key_fg], dim=2)
                saved_value = torch.cat([saved_value, saved_value_fg], dim=2)
                id_fg_mask = torch.zeros([saved_key.shape[2]], device=device, dtype=torch.bool)
                id_fg_mask[-saved_key_fg.shape[2]:] = True
            else:
                saved_key = saved_key_fg
                saved_value = saved_value_fg
                id_fg_mask = torch.ones([saved_key.shape[2]], device=device, dtype=torch.bool)
        
        expand_mask= self.get_expand_attn_mask(
            id_fg_mask, curr_fg_mask, bg_share_flag, fg_share_flag, device=device)
        attention_mask = torch.zeros(
            (1, 1, self.text_seq_len + self.visual_seq_len, self.text_seq_len + self.visual_seq_len + len(id_fg_mask)), 
            device=device, dtype=torch.bfloat16)
        attention_mask[:, :, :, self.text_seq_len + self.visual_seq_len:].masked_fill_(expand_mask, float('-inf'))
        return saved_key, saved_value, attention_mask
    
    
    def ada_tome(self, hidden_states, timestep_ind, alpha, id_fg_inds=None, curr_fg_inds=None, max_sim=None, **kwargs):
        id_hidden_states = self.id_attn_bank[timestep_ind]["attn_out"].to(hidden_states.device, non_blocking=True)
        vision_hidden_states = hidden_states[:, self.text_seq_len:, :]
        matched_id_hidden_states = id_hidden_states[:, id_fg_inds]
        matched_curr_hidden_states = vision_hidden_states[:, curr_fg_inds]

        alpha_tensor = torch.ones_like(curr_fg_inds, dtype=torch.bfloat16)
        alpha_tensor = alpha_tensor * alpha
        sim_weight = max_sim.flatten()[curr_fg_inds]
        alpha_tensor = alpha_tensor * sim_weight
        alpha_tensor = alpha_tensor.view(1, -1, 1)

        new_matched_curr_hidden_states = (1 - alpha_tensor) * matched_curr_hidden_states + alpha_tensor * matched_id_hidden_states
        vision_hidden_states[:, curr_fg_inds] = new_matched_curr_hidden_states
        out_hidden_states = torch.cat((hidden_states[:, :self.text_seq_len, :], vision_hidden_states), dim=-2)
        return out_hidden_states


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # new args
        timestep_ind: Optional[int] = None,
        save_attn_weight: bool = False,
        save_attn_kv: bool = False,
        update_attn_kv: bool = False,
        save_attn_out_for_sim: bool = False,
        save_attn_out_for_interpolate: bool = False,
        save_cross_sim: bool = False,
        fg_inter_img_attn: bool = False,
        bg_inter_img_attn: bool = False,
        attn_out_interpolate: bool = False,
        interpolate_weight_dict: dict = dict(),
        spatial_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        
        if save_attn_kv:
            if timestep_ind not in self.id_attn_bank:
                self.id_attn_bank[timestep_ind] = dict()
            self.id_attn_bank[timestep_ind]["key"] = key[:, :, self.text_seq_len:, :].cpu()
            self.id_attn_bank[timestep_ind]["value"] = value[:, :, self.text_seq_len:, :].cpu()
        
        if update_attn_kv:
            if timestep_ind not in self.id_attn_bank:
                self.id_attn_bank[timestep_ind] = dict()
            self.id_attn_bank[timestep_ind]["updated_key"] = key[:, :, self.text_seq_len:, :].cpu()
            self.id_attn_bank[timestep_ind]["updated_value"] = value[:, :, self.text_seq_len:, :].cpu()

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if save_attn_weight:
            bg_attn, fg_attn = self.get_curr_attn_weights(query, key)
            self.attn_weights = dict(bg = bg_attn.to("cuda:0", non_blocking=True), fg = fg_attn.to("cuda:0", non_blocking=True))
        
        fg_share_flag = self.fg_share_flag and fg_inter_img_attn
        bg_share_flag = self.bg_share_flag and bg_inter_img_attn and (not update_attn_kv)
        if fg_share_flag or bg_share_flag:
            saved_key, saved_value, attention_mask = self.get_expanded_key_value(
                (image_rotary_emb[0][self.text_seq_len:], image_rotary_emb[1][self.text_seq_len:]),
                bg_share_flag,
                fg_share_flag,
                timestep_ind,
                query.device,
                **spatial_kwargs)
            key = torch.cat([key, saved_key], dim=2)
            value = torch.cat([value, saved_value], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if save_attn_out_for_sim:
            if timestep_ind not in self.id_attn_bank:
                self.id_attn_bank[timestep_ind] = dict()
            self.id_attn_bank[timestep_ind]["attn_out"] = hidden_states[:, self.text_seq_len:, :].cpu()
        elif save_attn_out_for_interpolate and self.fg_share_flag:
            if timestep_ind not in self.id_attn_bank:
                self.id_attn_bank[timestep_ind] = dict()
            self.id_attn_bank[timestep_ind]["attn_out"] = hidden_states[:, self.text_seq_len:, :].cpu()

        if save_cross_sim:
            self.cross_sims = self.get_curr_cross_sim(hidden_states[:, self.text_seq_len:, :], timestep_ind).to("cuda:0", non_blocking=True)

        if attn_out_interpolate:
            if fg_share_flag:
                hidden_states = self.ada_tome(hidden_states, timestep_ind, interpolate_weight_dict[timestep_ind], **spatial_kwargs)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def reset_attn_processor(pipe, size, fg_share_freq=2, bg_share_freq=1):
    new_attn_processors = dict()
    transformer = pipe.transformer
    ori_attn_processors = transformer.attn_processors
    reset_num = 0
    for name in ori_attn_processors:
        if 'single' in name:
            block_ind = int(name.split(".")[1])
            new_attn_processors[name] = CharaConsistAttnProcessor2_0(
                size=size,
                fg_share_flag=(block_ind % fg_share_freq == 0),
                bg_share_flag=(block_ind % bg_share_freq == 0),
            )
            print(f"reset attn processor of layer {name}")
            reset_num += 1
        else:
            new_attn_processors[name] = FluxAttnProcessor2_0()
    transformer.set_attn_processor(new_attn_processors)
    print(f"{reset_num} layers have been reset")


def set_text_len(pipe, bg_len, real_len):
    attn_processors = pipe.transformer.attn_processors
    reset_num = 0
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            processor.bg_len = bg_len
            processor.real_len = real_len
            reset_num += 1
    print(f"{reset_num} layers' background and real text length have been reset to {bg_len} and {real_len}.")

def remove_small_holes_and_points(mask_tensor):
    n, h, w = mask_tensor.shape
    results = []
    for i in range(n):
        mask = mask_tensor[i].cpu().numpy().astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        results.append(torch.tensor(mask))
    results = torch.stack(results, dim=0).to(mask_tensor.device, dtype=mask_tensor.dtype)
    return results

def get_curr_fg_mask(pipe):
    attn_processors = pipe.transformer.attn_processors
    all_attn_weights = dict(bg = [], fg = [])
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            saved_attns = processor.attn_weights
            for k in saved_attns:
                all_attn_weights[k].append(saved_attns[k])
            processor.attn_weights = dict()
    bg_attns = sum(all_attn_weights["bg"]) / len(all_attn_weights["bg"])
    fg_attns = sum(all_attn_weights["fg"]) / len(all_attn_weights["fg"])
    return remove_small_holes_and_points(bg_attns <= fg_attns)

def get_cross_sim(pipe):
    attn_processors = pipe.transformer.attn_processors
    all_cross_sims = []
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            all_cross_sims.append(processor.cross_sims)
            processor.cross_sims = None
    return sum(all_cross_sims) / len(all_cross_sims)

def reset_id_bank(pipe):
    attn_processors = pipe.transformer.attn_processors
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            processor.id_attn_bank = dict()
    gc.collect()

def reset_size(pipe, h, w):
    attn_processors = pipe.transformer.attn_processors
    for name in attn_processors:
        processor = attn_processors[name]
        if isinstance(processor, CharaConsistAttnProcessor2_0):
            processor.size = (h//16, w//16)
            processor.visual_seq_len = processor.size[0] * processor.size[1]
