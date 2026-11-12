"""
Train DDPM on BONBID2023 using only axial slices listed in ``axial_slices_no_lesion``.

Uses ``datasets/bonbid2023.BONBID2023HealthySliceDataset`` (image channels only).
"""

import copy
import logging
import os

import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import optim

import wandb
from datasets.bonbid2023 import bonbid2023_train_dataloader
from diffusion import *
from modules import *
from utils import *

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

    dataloader = bonbid2023_train_dataloader(conf)
    steps_per_epoch = int(np.ceil(len(dataloader.dataset) / conf["batch_size"]))
    number_of_steps = steps_per_epoch * conf["epochs"]

    model = UNet(
        c_in=conf["channels"],
        c_out=conf["channels"],
        device=device,
        img_size=conf["size"],
    )
    optimizer = optim.AdamW(model.parameters(), lr=1)
    mse = nn.MSELoss()
    diffusion = Diffusion(
        noise_steps=conf["noise_steps"],
        img_size=conf["size"],
        beta_start=conf["beta_start"],
        beta_end=conf["beta_end"],
        device=device,
    )
    ema = EMA(conf["ema_decay"])
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    if conf.get("train_continue"):
        conf["target_lr"] = conf["start_lr"]
        ckpt = torch.load(conf["model"])
        model.load_state_dict(ckpt)
        ckpt = torch.load(conf["opt"])
        optimizer.load_state_dict(ckpt)
        ckpt = torch.load(conf["ema"])
        ema_model.load_state_dict(ckpt)
        ema_model.eval().requires_grad_(False)

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
        logging.info(f"BONBID DDPM epoch {epoch}:")
        pbar = tqdm(dataloader) if accelerator.is_main_process else dataloader
        for i, images in enumerate(pbar):
            images = (images * 2) - 1
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(
                images,
                t,
                simplex=conf["simplex"],
                pyramid=conf["pyramid"],
                discount=conf["discount"],
            )
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            ema.step_ema(ema_model, model)
            if accelerator.is_main_process:
                wandb.log({"MSE": loss.item()})
                pbar.set_description(f"Loss: {loss.item():.4f}")

        if (
            epoch > conf["start_ckpt"]
            and epoch % conf["save_ckpt"] == 0
            and accelerator.is_main_process
        ):
            my_model = accelerator.unwrap_model(model)
            my_ema_model = accelerator.unwrap_model(ema_model)
            accelerator.save(
                my_model.state_dict(),
                os.path.join("models", conf["run_name"], f"{epoch}_ckpt.pt"),
            )
            accelerator.save(
                optimizer.state_dict(),
                os.path.join("models", conf["run_name"], f"{epoch}_optim.pt"),
            )
            ema_sampled_images = diffusion.sample(
                my_ema_model,
                n=conf["num_images"],
                channels=conf["channels"],
                pyramid=conf["pyramid"],
                simplex=conf["simplex"],
                discount=conf["discount"],
            )
            save_images(
                ema_sampled_images,
                os.path.join("results", conf["run_name"], f"{epoch}_ema.jpg"),
                mode="L",
            )
            accelerator.save(
                my_ema_model.state_dict(),
                os.path.join("models", conf["run_name"], f"{epoch}_ema_ckpt.pt"),
            )
            wandb.log(
                {
                    "EMA-DDPM": wandb.Image(
                        upload_images(ema_sampled_images, mode="L")
                    )
                }
            )


def main():
    torch.backends.cudnn.benchmark = True
    with open("./train_bonbid_affine_double_space.yml", "r") as f:
        conf = yaml.safe_load(f)
    wandb.init(
        entity=conf.get("wandb_entity", "pkan2-"),
        project=conf["run_name"],
        config=conf,
    )
    train(conf)


if __name__ == "__main__":
    main()
