"""
Train ControlNet on BONBID2023 healthy slices only, with separate control channels (e.g. Z-map).

Expects ``train_controlnet_bonbid.yml`` with ``dataset_channels``, ``control_columns``,
and the same CSV format as ``datasets/bonbid2023.py` (``axial_slices_no_lesion``).
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import optim

import wandb
from control_net import ControlNetUNet
from datasets.bonbid2023 import bonbid2023_control_dataloader
from diffusion import Diffusion
from modules import EMA, LRWarmupCosineDecay, UNet
from utils import make_dicts, save_images, upload_images

logging.basicConfig(format="%(message)s", level=logging.INFO)
os.environ["WANDB__SERVICE_WAIT"] = "300"


def train(conf):
    make_dicts(conf["run_name"])
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device

    if conf.get("simplex") and conf.get("pyramid"):
        conf["simplex"] = False
        conf["pyramid"] = False

    dataloader = bonbid2023_control_dataloader(conf)

    steps_per_epoch = int(np.ceil(len(dataloader.dataset) / conf["batch_size"]))
    number_of_steps = steps_per_epoch * conf["epochs"]

    base_template = UNet(
        c_in=conf["channels"],
        c_out=conf["channels"],
        device=device,
        img_size=conf["size"],
    )
    if not conf.get("control_columns"):
        raise ValueError("control_columns required in config (e.g. zmap_affine).")
    control_ch = len(conf["control_columns"])
    model = ControlNetUNet(
        base_template,
        control_channels=control_ch,
        img_size=conf["size"],
    )
    pretrained = conf["pretrained_unet_path"]
    if not os.path.isfile(pretrained):
        raise FileNotFoundError(f"pretrained_unet_path not found: {pretrained}")
    model.load_base_from_checkpoint(
        pretrained, map_location="cpu", strict=conf.get("load_base_strict", True)
    )

    trainable = [
        p for n, p in model.named_parameters() if not n.startswith("base.")
    ]
    optimizer = optim.AdamW(trainable, lr=1.0)
    mse = nn.MSELoss()
    diffusion = Diffusion(
        noise_steps=conf["noise_steps"],
        img_size=conf["size"],
        beta_start=conf["beta_start"],
        beta_end=conf["beta_end"],
        device=device,
    )
    ema = EMA(conf["ema_decay"])
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.base.parameters():
        p.requires_grad = False

    cfg_dropout = float(conf.get("cfg_dropout", 0.1))
    pyramid = conf.get("pyramid", True)
    discount = conf.get("discount", 0.8)
    simplex = conf.get("simplex", False)

    if conf.get("train_continue"):
        conf["target_lr"] = conf["start_lr"]
        ckpt = torch.load(conf["controlnet_ckpt"], map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        if conf.get("controlnet_opt"):
            optimizer.load_state_dict(
                torch.load(conf["controlnet_opt"], map_location="cpu")
            )
        if conf.get("controlnet_ema"):
            ema_model.load_state_dict(
                torch.load(conf["controlnet_ema"], map_location="cpu")
            )

    scheduler = LRWarmupCosineDecay(
        optimizer,
        int(conf["warmup_steps"] * number_of_steps),
        number_of_steps,
        conf["start_lr"],
        conf["target_lr"],
    )
    model, ema_model, optimizer, scheduler, dataloader = accelerator.prepare(
        model, ema_model, optimizer, scheduler, dataloader
    )

    from tqdm import tqdm

    for epoch in range(conf["epochs"]):
        logging.info(f"BONBID ControlNet epoch {epoch}:")
        pbar = tqdm(dataloader) if accelerator.is_main_process else dataloader
        model.train()
        last_control = None
        for batch in pbar:
            if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
                raise ValueError(
                    "Expected (images, control) batches from bonbid2023_control_dataloader."
                )
            images, control = batch
            images = (images * 2) - 1
            control = control.to(device)
            last_control = control.detach()
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(
                images,
                t,
                simplex=simplex,
                pyramid=pyramid,
                discount=discount,
            )
            predicted_noise = model(
                x_t, t, control=control, cfg_dropout_prob=cfg_dropout
            )
            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            ema.step_ema(ema_model, model)
            if accelerator.is_main_process:
                wandb.log({"controlnet_mse": loss.item()})
                pbar.set_description(f"Loss: {loss.item():.4f}")

        if (
            last_control is not None
            and epoch > conf["start_ckpt"]
            and epoch % conf["save_ckpt"] == 0
            and accelerator.is_main_process
        ):
            my_model = accelerator.unwrap_model(model)
            my_ema = accelerator.unwrap_model(ema_model)
            out_dir = os.path.join("models", conf["run_name"])
            accelerator.save(
                my_model.state_dict(),
                os.path.join(out_dir, f"{epoch}_controlnet_full.pt"),
            )
            accelerator.save(
                {
                    k: v
                    for k, v in my_model.state_dict().items()
                    if not k.startswith("base.")
                },
                os.path.join(out_dir, f"{epoch}_controlnet_trainable.pt"),
            )
            accelerator.save(
                optimizer.state_dict(),
                os.path.join(out_dir, f"{epoch}_controlnet_optim.pt"),
            )
            accelerator.save(
                my_ema.state_dict(),
                os.path.join(out_dir, f"{epoch}_controlnet_ema.pt"),
            )
            cfg_scale = float(conf.get("cfg_scale_sample", 7.5))
            ni = conf["num_images"]
            ctrl = last_control[:ni].clone()
            if ctrl.shape[0] < ni:
                pad = ni - ctrl.shape[0]
                ctrl = torch.cat(
                    [ctrl, last_control[:1].expand(pad, *last_control.shape[1:])],
                    dim=0,
                )
            ema_sampled = diffusion.sample_controlnet(
                my_ema,
                n=conf["num_images"],
                channels=conf["channels"],
                control=ctrl,
                cfg_scale=cfg_scale,
                pyramid=pyramid,
                simplex=simplex,
                discount=discount,
            )
            save_images(
                ema_sampled,
                os.path.join(
                    "results", conf["run_name"], f"{epoch}_controlnet_ema.jpg"
                ),
                mode="L",
            )
            wandb.log(
                {
                    "ControlNet-sample": wandb.Image(
                        upload_images(ema_sampled, mode="L")
                    )
                }
            )


def main():
    torch.backends.cudnn.benchmark = True
    with open("./train_controlnet_bonbid_affine_double_space.yml", "r") as f:
        conf = yaml.safe_load(f)
    wandb.init(
        entity=conf.get("wandb_entity", "pkan2-"),
        project=conf["run_name"],
        config=conf,
    )
    train(conf)


if __name__ == "__main__":
    main()
