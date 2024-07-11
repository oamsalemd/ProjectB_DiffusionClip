import matplotlib.pyplot as plt
import diffusers
from diffusers import *
import torch
from glob import glob
from diffusers.utils import load_image
from torchvision import transforms
from diffusers.utils import torch_utils
import numpy as np
from typing import List, Optional, Tuple, Union
import argparse

# os.chdir("/home/ohada/ProjectBDir")

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
    # Define the transformation
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
    # noise tensor
    noise_encoded = torch_utils.randn_tensor(img_encoded.shape, dtype=img_encoded.dtype).to(device)

    return ((alpha ** 0.5) * img_encoded) + (((1 - alpha) ** 0.5) * noise_encoded)


def unclip(num_inference_steps=20, eta=0, dataset=('horse', 'zebra'), test_idx=0):
    # Load the model
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
    )

    # Load the *modified* scheduler
    pipe.scheduler = DDIMScheduler_alpha.from_config(pipe.scheduler.config)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.config.timestep_spacing = "linspace"
    pipe = pipe.to("cuda")

    with torch.no_grad():
        # Load the images
        from_images = [load_image(path) for path in sorted(glob("datasets/{source}*.jpg".format(source=dataset[0])))]
        to_images = [load_image(path) for path in sorted(glob("datasets/{to}*.jpg".format(to=dataset[1])))]
        images = from_images + to_images

        # Encode the images
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

        # Extract test image
        ds_to_embeds = torch.cat((to_embeds[0:test_idx], to_embeds[test_idx + 1:]), dim=0)
        ds_from_embeds = torch.cat((from_embeds[0:test_idx], from_embeds[test_idx + 1:]), dim=0)

        # Calculate the difference between the embeddings (excluding the test image)
        diff = (ds_to_embeds - ds_from_embeds).mean(0)

        # Encode the test image
        scaled_latents = encode_img(pipe, from_images[test_idx], alpha=start_alpha, device="cuda")

        # Run the pipeline
        output_to = pipe(
            num_inference_steps=num_inference_steps,
            latents=scaled_latents,
            eta=eta,
            image_embeds=(from_embeds[test_idx] + diff)[None, :1024]
        ).images

    # Save the output image
    # output_to[0].save(
    #     f"variation_image_steps={num_inference_steps}_alpha={start_alpha}.png")
    return output_to[0]


def sweep_images(num_inference_steps=(20,), alpha=(0,), eta=0, dataset=('horse', 'zebra'), test_idx=0):
    # Create a figure with subplots
    fig, axs = plt.subplots(len(num_inference_steps), len(alpha), figsize=(20*len(alpha), 20*len(num_inference_steps)))

    # If only one value is given for num_inference_steps and alpha, convert them to lists
    if len(num_inference_steps) == 1 and len(alpha) == 1:
        axs = np.array([[axs]])
    elif len(num_inference_steps) == 1:
        axs = np.array([[axs[i] for i in range(len(alpha))]])
    elif len(alpha) == 1:
        axs = np.array([[axs[i]] for i in range(len(num_inference_steps))])
    else:
        axs = np.array([[axs[i][j] for j in range(len(alpha))] for i in range(len(num_inference_steps))])

    # Loop over the subplots and plot the images
    for i, nis in enumerate(num_inference_steps):
        for j, a in enumerate(alpha):
            global start_alpha
            start_alpha = a
            img = unclip(nis, eta, dataset, test_idx)
            axs[i, j].imshow(img)
            axs[i, j].axis("off")
            axs[i, j].set_title(f"Steps={nis}, Alpha={a}")

    # Add a title to the figure
    fig.suptitle(f"Variation of UnCLIP with different number of steps and alpha values\nDataset: {dataset}, Test image index: {test_idx}, Eta={eta}", fontsize=36)

    # Save the figure
    img_name = f"sweep_result--steps={num_inference_steps}_alpha={alpha}_eta={eta}_dataset={dataset}_test_idx={test_idx}.png"
    plt.savefig(img_name)

    # Print the name of the saved image
    print(f"\n=== Saved image as '{img_name}'")


if __name__ == "__main__":
    # img = unclip(dataset=('door_close', 'door_open'))
    # sweep_images(num_inference_steps=(20,), alpha=(0,), eta=0, dataset=('horse', 'zebra'), test_idx=0)

    # Take arguments from the command line:
    argsp = argparse.ArgumentParser()
    argsp.add_argument("--num_inference_steps", nargs='+', type=int, default=20, help="Number of inference steps to sweep over")
    argsp.add_argument("--alpha", type=float, nargs='+', default=0, help="Alpha values to sweep over")
    argsp.add_argument("--eta", type=float, default=0, help="Eta value")
    argsp.add_argument("--dataset", type=str, default="1", choices=("1", "2", "3"), help="1: horse2zebra, 2: open2close, 3: black2blue")
    argsp.add_argument("--test_idx", type=int, default=0, help="Index of test image out of the dataset")
    args = argsp.parse_args()

    # Parse arguments:
    parsed_num_inference_steps = args.num_inference_steps if isinstance(args.num_inference_steps, list) else [args.num_inference_steps]
    parsed_alpha = args.alpha if isinstance(args.alpha, list) else [args.alpha]
    parsed_dataset = ("horse", "zebra") if args.dataset == "1" else ("door_close", "door_open") if args.dataset == "2" else ("shirt_black", "shirt_blue")

    # Print parsed arguments:
    print("\n=== Parsed arguments:")
    print(f"num_inference_steps: {parsed_num_inference_steps}")
    print(f"alpha: {parsed_alpha}")
    print(f"eta: {args.eta}")
    print(f"dataset: {parsed_dataset}")
    print(f"test_idx: {args.test_idx}\n")

    # Run the sweep_images function with the parsed arguments:
    sweep_images(num_inference_steps=tuple(parsed_num_inference_steps), alpha=tuple(parsed_alpha), eta=args.eta, dataset=parsed_dataset, test_idx=args.test_idx)