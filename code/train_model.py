import argparse
import logging
import math
import os
from pathlib import Path
import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import datetime
from mask_encoder import ControlNetConditioningEmbedding
from accelerate import DistributedDataParallelKwargs

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from training_cfg import load_training_config
check_min_version("0.26.0.dev0")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    args = parser.parse_args()
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600000)) 

    cfg_path = args.cfg

    cfg = load_training_config(cfg_path)
    logging_dir = Path(cfg.log_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, logging_dir=logging_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    print(accelerator.project_dir)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if cfg.seed is not None:
        set_seed(cfg.seed)

    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    mask_encoder = ControlNetConditioningEmbedding()

    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet"
    )
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules= r"down_blocks\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.to_(k|q|v|out\.0)|"
                        r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.to_(k|q|v|out\.0)|"
                        r"up_blocks\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.to_(k|q|v|out\.0)"
    )
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    mask_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.add_adapter(unet_lora_config)

    if cfg.mixed_precision == "fp16":
        for param in unet.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)
        for param in mask_encoder.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        list(lora_layers) + list(mask_encoder.parameters()),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    cfg_data = OmegaConf.load("./latent_LayoutDiffusion_large.yaml")
    train_dataset = make_train_dataset_grit(cfg_data, 'train', accelerator, tokenizer)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer
    )

    unet, optimizer, train_dataloader, lr_scheduler,mask_encoder = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler,mask_encoder
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    first_epoch = 0

    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, cfg.num_train_epochs):
        unet.train()
        mask_encoder.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([unet, mask_encoder]):
                
                batch_images, batch_cond = batch

                latents = vae.encode(batch_images.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                if cfg.noise_offset:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                bsize = len(batch_cond['obj_class_text_ids']) 
                cond_mask = []
                encoder_hidden_states_batch = []

                for cid in range(bsize):
                    bsid = min(batch_cond["num_selected"][cid], 5)
                    encoder_hidden_states = text_encoder(batch_cond['obj_class_text_ids'][cid])[0][1:bsid]

                    controlnet_image = batch_cond["cond_image"][cid].to(dtype=weight_dtype)[1:bsid]
                    mask_cond = mask_encoder(controlnet_image)
                    cond_mask.append(mask_cond)
                    encoder_hidden_states_batch.append(encoder_hidden_states)

                unet_encoder_hidden_states = text_encoder(torch.stack(batch_cond["base_class_text_ids"]).squeeze(1))[0]
                model_pred = unet(noisy_latents, timesteps,unet_encoder_hidden_states,cond_mask =cond_mask,encoder_hidden_states_batch=encoder_hidden_states_batch ).sample

                loss = F.mse_loss(model_pred.float(),
                                  target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(unet.parameters()) + list(mask_encoder.parameters())
                    accelerator.clip_grad_norm_(
                        params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        unwrap_mask_encoder = accelerator.unwrap_model(mask_encoder)
                        unwrap_mask_encoder_state_dict = unwrap_mask_encoder.state_dict()
                        mask_encoder_save_path = os.path.join(save_path, "mask_encoder.pth")
                        torch.save(unwrap_mask_encoder_state_dict, mask_encoder_save_path)


            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=cfg.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
            weight_name=cfg.ckpt_name + '.safetensor'
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
