import os

import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('GTK3Agg')
import optuna
import diffusers
from diffusers import *
import torch
from glob import glob
from diffusers.utils import load_image
from torchvision import transforms
from diffusers.utils import torch_utils
import numpy as np
from typing import List, Optional, Tuple, Union

os.chdir("/home/ohada/ProjectBDir")

start_alpha = 0.0


class DDIMScheduler_alpha(DDIMScheduler):
    def __init__(self):
        super().__init__()

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """

        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            end_timestep = self.config.num_train_timesteps - 1
            for timestep, alpha in enumerate(self.alphas_cumprod):
                if alpha < start_alpha:
                    end_timestep = timestep
                    break
            timesteps = (
                np.linspace(0, end_timestep, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = torch.from_numpy(timesteps).to(device)


def encode_img(pipe, img, alpha=0.0, device="cuda"):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.half()),
        # Resize to 768x768:
        transforms.Resize((768, 768))
    ]
    )

    # image to tensor
    t_img = to_tensor(img).to(device)
    # tensor pre-process
    t_img = ((t_img - 0.5) * 2)
    # tensor to latents
    img_encoded = pipe.vae.encode(t_img.unsqueeze(0), return_dict=False)[0].mean
    # scaling latent
    img_encoded = img_encoded * pipe.vae.config.scaling_factor

    noise_encoded = torch_utils.randn_tensor(img_encoded.shape, dtype=img_encoded.dtype).to(device)

    return ((alpha ** 0.5) * img_encoded) + (((1 - alpha) ** 0.5) * noise_encoded)


def unclip(num_inference_steps=20, eta=0, dataset=('horse', 'zebra')):
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
    )
    pipe.scheduler = DDIMScheduler_alpha.from_config(pipe.scheduler.config)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.config.timestep_spacing = "linspace"
    pipe = pipe.to("cuda")

    with torch.no_grad():
        from_images = [load_image(path) for path in sorted(glob("datasets/{source}*.jpg".format(source=dataset[0])))]
        to_images = [load_image(path) for path in sorted(glob("datasets/{to}*.jpg".format(to=dataset[1])))]
        images = from_images + to_images
        image_embeds = [pipe._encode_image(
            image=(im,),
            device=pipe.device,
            batch_size=len(images),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            noise_level=torch.zeros((1,), device="cuda", dtype=torch.int),
            generator=None,
            image_embeds=None,
        ) for im in images]
        from_embeds, to_embeds = torch.cat(image_embeds).chunk(2, dim=0)
        diff = (to_embeds[1:] - from_embeds[1:]).mean(0)

        scaled_latents = encode_img(pipe, from_images[0], alpha=start_alpha, device="cuda")

        output_to = pipe(
            num_inference_steps=num_inference_steps,
            latents=scaled_latents,
            eta=eta,
            image_embeds=(from_embeds[0] + diff)[None, :1024]
        ).images

    output_to[0].save(
        f"variation_image_steps={num_inference_steps}_alpha={start_alpha}.png")
    return output_to[0]


if __name__ == "__main__":
    start_alpha = 0.1
    img = unclip(dataset=('door_close', 'door_open'))
