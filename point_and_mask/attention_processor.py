# Copied and modified from diffusers/blob/main/src/diffusers/models/attention_processor.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
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
        save_attn_weight: bool = False,
        save_attn_out: bool = False,
        save_cross_sim: bool = False,
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


class MaskPointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.bg_len = 0
        self.real_len = 0
        self.size = (64, 64)
        self.text_seq_len = 512
        self.visual_seq_len = 64 * 64
        self.id_attn_bank = None
        self.attn_weights = dict()
        self.cross_sims = None

    def get_curr_attn_weights(self, q, k):
        attn_weights = torch.matmul(q[:, :, self.text_seq_len:], k[:, :, :self.real_len].transpose(-2, -1)).to(torch.float16)
        attn_weights = attn_weights.softmax(-1).sum(1)
        attn_weights = rearrange(attn_weights, "b (h w) s -> b s h w", h=self.size[0], w=self.size[1])
        bg_attn_weights = attn_weights[:, :self.bg_len].mean(1)
        fg_attn_weights = attn_weights[:, self.bg_len:].mean(1)       
        return bg_attn_weights, fg_attn_weights
    
    def get_curr_cross_sim(self, curr_hidden_states):
        id_hidden_states = self.id_attn_bank.to(curr_hidden_states.device, non_blocking=True)
        sim = torch.matmul(
            F.normalize(curr_hidden_states, dim=-1),
            F.normalize(id_hidden_states, dim=-1).transpose(-2, -1))
        return sim

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # new args
        save_attn_weight: bool = False,
        save_attn_out: bool = False,
        save_cross_sim: bool = False,
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

        if save_attn_weight:
            bg_attn, fg_attn = self.get_curr_attn_weights(query, key)
            self.attn_weights = dict(bg = bg_attn.to("cuda:0", non_blocking=True), fg = fg_attn.to("cuda:0", non_blocking=True))
        
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if save_attn_out:
            self.id_attn_bank = hidden_states[:, self.text_seq_len:, :].cpu()

        if save_cross_sim:
            self.cross_sims = self.get_curr_cross_sim(hidden_states[:, self.text_seq_len:, :]).to("cuda:0", non_blocking=True)

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


def reset_attn_processor(pipe):
    new_attn_processors = dict()
    transformer = pipe.transformer
    ori_attn_processors = transformer.attn_processors
    reset_num = 0
    for name in ori_attn_processors:
        if 'single' in name:
            new_attn_processors[name] = MaskPointAttnProcessor2_0()
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
        if isinstance(processor, MaskPointAttnProcessor2_0):
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
        if isinstance(processor, MaskPointAttnProcessor2_0):
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
        if isinstance(processor, MaskPointAttnProcessor2_0):
            all_cross_sims.append(processor.cross_sims)
            processor.cross_sims = None
    return sum(all_cross_sims) / len(all_cross_sims)
