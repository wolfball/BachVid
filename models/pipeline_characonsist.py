# Copied and modified from diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import copy

from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from .attention_processor_characonsist import get_curr_fg_mask, get_cross_sim


def get_interpolate_weight(weight, start_step, decay_step, end_step):
    steps = np.arange(0, end_step - decay_step)
    decay_weights = weight * 0.5 * (1 + np.cos(np.pi * steps / (end_step - decay_step)))
    decay_weights = decay_weights.tolist()
    constant_weights = [weight] * (decay_step - start_step)
    weight_list = constant_weights + decay_weights
    weight_dict = dict()
    for ind, interpolate_step in enumerate(range(start_step, end_step)):
        weight_dict[interpolate_step] = weight_list[ind]
    return weight_dict

def get_shared_fg_mask(id_fg_mask, curr_fg_mask, curr2id_argmax_indices, curr2id_valid_mask):
    curr_valid_mask = curr2id_valid_mask.flatten()
    id_fg_mask = id_fg_mask.flatten().to(curr2id_argmax_indices.device)
    curr_fg_mask = curr_fg_mask.flatten().to(curr2id_argmax_indices.device)
    rearrange_id_fg_mask = id_fg_mask[curr2id_argmax_indices[0]]
    share_fg_mask = curr_fg_mask & rearrange_id_fg_mask & curr_valid_mask
    id_share_fg_indices = curr2id_argmax_indices[0][share_fg_mask]
    curr_share_fg_indices = torch.nonzero(share_fg_mask).squeeze()
    return id_share_fg_indices, curr_share_fg_indices


class CharaConsistPipeline(FluxPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = False,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # extra args
        spatial_kwargs: dict = dict(),
        is_id: bool = False,
        is_pre_run: bool = False,
        use_interpolate: bool = True,
        share_bg: bool = True,
        update_bg: bool = False,
        attn_start_step: int = 1,
        attn_end_step: int = 41,
        interpolate_start_step: int = 1,
        interpolate_decay_step: int = 11,
        interpolate_end_step: int = 31,
        interpolate_weight: float = 0.8,
        sim_thr = 0.5,
        save_mask_point_step: int = 10
    ):
        
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        interpolate_weight_dict = get_interpolate_weight(
            interpolate_weight, interpolate_start_step, interpolate_decay_step, interpolate_end_step)

        def get_consist_kwargs(i):
            if is_id:
                save_attn_weight = i == save_mask_point_step
                save_attn_kv = (i < attn_end_step) and (i >= attn_start_step)
                update_attn_kv = update_bg & (i < attn_end_step) and (i >= attn_start_step)
                save_attn_out_for_sim = i == save_mask_point_step
                save_attn_out_for_interpolate = use_interpolate and (i < interpolate_end_step) and (i >= interpolate_start_step)
                save_cross_sim = False
                fg_inter_img_attn = False
                bg_inter_img_attn = False
                attn_out_interpolate = False
            elif is_pre_run:
                save_attn_weight = i <= save_mask_point_step
                save_attn_kv = False
                update_attn_kv = False
                save_attn_out_for_sim = False
                save_attn_out_for_interpolate = False
                save_cross_sim = i == save_mask_point_step
                fg_inter_img_attn = False
                bg_inter_img_attn = (i < attn_end_step) and (i >= attn_start_step) and share_bg and (not update_bg)
                attn_out_interpolate = False
            else:
                save_attn_weight = i <= save_mask_point_step
                save_attn_kv = False
                update_attn_kv = update_bg & (i < attn_end_step) and (i >= attn_start_step)
                save_attn_out_for_sim = False
                save_attn_out_for_interpolate = False
                save_cross_sim = i == save_mask_point_step
                fg_inter_img_attn = (i < attn_end_step) and (i >= attn_start_step)
                bg_inter_img_attn = (i < attn_end_step) and (i >= attn_start_step) and share_bg and (not update_bg)
                attn_out_interpolate = use_interpolate & (i < interpolate_end_step) and (i >= interpolate_start_step)
            return dict(
                timestep_ind=i,
                save_attn_weight = save_attn_weight,
                save_attn_kv = save_attn_kv,
                update_attn_kv=update_attn_kv,
                save_attn_out_for_sim = save_attn_out_for_sim,
                save_attn_out_for_interpolate = save_attn_out_for_interpolate,
                save_cross_sim = save_cross_sim,
                fg_inter_img_attn = fg_inter_img_attn,
                bg_inter_img_attn = bg_inter_img_attn,
                attn_out_interpolate = attn_out_interpolate,
                interpolate_weight_dict=interpolate_weight_dict,
                spatial_kwargs=spatial_kwargs)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._joint_attention_kwargs = get_consist_kwargs(i)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.joint_attention_kwargs["save_attn_weight"]:
                    curr_fg_mask = get_curr_fg_mask(self)
                    spatial_kwargs["curr_fg_mask"] = curr_fg_mask
                    if update_bg:
                        spatial_kwargs["id_bg_mask"] = copy.deepcopy(~curr_fg_mask)
                        
                if self.joint_attention_kwargs["save_cross_sim"]:
                    avg_cross_sim = get_cross_sim(self)
                    max_sim, argmax_indices = torch.max(avg_cross_sim, dim=-1)
                    id_fg_inds, curr_fg_inds = get_shared_fg_mask(
                        spatial_kwargs["id_fg_mask"], 
                        spatial_kwargs["curr_fg_mask"], 
                        argmax_indices,
                        max_sim>sim_thr
                    )
                    spatial_kwargs.update(
                        id_fg_inds = id_fg_inds,
                        curr_fg_inds = curr_fg_inds,
                        max_sim=max_sim,
                        argmax_indices=argmax_indices, 
                    )
                
                if is_pre_run and (i == save_mask_point_step):
                    latents = (latents - self.scheduler.sigmas[i] * noise_pred)
                    break
                
                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, spatial_kwargs)

        return FluxPipelineOutput(images=image)


