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

os.chdir("/home/ohada/ProjectBDir")

def encode_img(pipe, img, scale=1, device="cuda"):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.half()),
        # Resize to 768x768:
        transforms.Resize((768, 768))
    ]
    )

    t_img = to_tensor(img).to(device)
    img_encoded = pipe.vae.encode(t_img.unsqueeze(0), return_dict=False)[0].mean
    noise_encoded = torch_utils.randn_tensor(img_encoded.shape, dtype=img_encoded.dtype).to(device)

    return scale * img_encoded + (1-scale) * noise_encoded


def unclip(num_inference_steps=20, guidance_scale=10.0, noise_level=0, latents=None):
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
    )
    pipe = pipe.to("cuda")
    with torch.no_grad():
        horse_images = [load_image(path) for path in sorted(glob("horse2zebra/horse*.jpg"))]
        zebra_images = [load_image(path) for path in sorted(glob("horse2zebra/zebra*.jpg"))]
        images = horse_images + zebra_images
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
        horse_embeds, zebra_embeds = torch.cat(image_embeds).chunk(2, dim=0)
        diff = (zebra_embeds[1:] - horse_embeds[1:]).mean(0)

        images_zebras = pipe(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            latents=latents,
            image_embeds=(horse_embeds[0]+diff)[None, :1024]).images

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.half())
        ]
    )

    images_zebras[0].save(f"variation_image_n={num_inference_steps}_g={guidance_scale:.2f}_n={noise_level}.png")
    return images_zebras[0]

if __name__ == "__main__":
    img = unclip()