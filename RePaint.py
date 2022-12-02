#!/usr/bin/env python

from io import BytesIO

import torch

import PIL
import requests
from diffusers import RePaintPipeline, RePaintScheduler
from pycomar.images import show3plt
import gradio as gr
import numpy as np


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


def auto():
    img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba_hq_256.png"
    mask_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask_256.png"

    # Load the original image and the mask as PIL images
    original_image = download_image(img_url).resize((256, 256))
    mask_image = download_image(mask_url).resize((256, 256))

    # Load the RePaint scheduler and pipeline based on a pretrained DDPM model
    scheduler = RePaintScheduler.from_pretrained(
        "google/ddpm-ema-celebahq-256")
    pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256",
                                           scheduler=scheduler)
    pipe = pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(0)
    output = pipe(
        original_image=original_image,
        mask_image=mask_image,
        num_inference_steps=250,
        eta=0.0,
        jump_length=10,
        jump_n_sample=10,
        generator=generator,
    )
    inpainted_image = output.images[0]

    show3plt([original_image, mask_image, inpainted_image])


class System(object):

    def __init__(self):

        # Load the RePaint scheduler and pipeline based on a pretrained DDPM model
        scheduler = RePaintScheduler.from_pretrained(
            "google/ddpm-ema-celebahq-256")
        pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256",
                                               scheduler=scheduler)
        self.pipe = pipe.to("cuda")
        self.generator = torch.Generator(device="cuda").manual_seed(0)

        # Construct UI
        img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba_hq_256.png"
        original_image = download_image(img_url).resize((256, 256))
        with gr.Blocks() as self.demo:
            with gr.Box():
                btn_get_mask = gr.Button("Get Mask")
                with gr.Row():
                    mask = gr.ImageMask(label="Inpaining")
                    view_original = gr.Image(label="Original", interactive=False)
                    view_mask = gr.Image(label="Mask", interactive=False)
                with gr.Row():
                    num_step = gr.Slider(minimum=1, maximum=250, value=250, label="The number of step")
                    jump_len = gr.Slider(minimum=1, maximum=20, value=10, label="Jump length")
                    jump_n_sam = gr.Slider(minimum=1, maximum=20, value=10, label="Jump N sample")
            with gr.Box():
                btn_inpaint = gr.Button("Inpaint")
                view_result = gr.Gallery(label="Result",
                                         interactive=False).style(grid=3)
            # PRELOAD
            self.demo.load(lambda: original_image, outputs=mask)
            btn_get_mask.click(lambda x: x["image"],
                               inputs=mask,
                               outputs=view_original)
            btn_get_mask.click(lambda x: 255 - x["mask"][..., :3],
                               inputs=mask,
                               outputs=view_mask)
            btn_inpaint.click(self.inference,
                              inputs=[
                                  view_original, view_mask, num_step, jump_len,
                                  jump_n_sam
                              ],
                              outputs=view_result)

    def launch(self):
        self.demo.launch()

    def inference(self, img: np.ndarray, mask: np.ndarray, num_step,
                  jump_length, jump_n_sample):
        img = PIL.Image.fromarray(img)
        mask = PIL.Image.fromarray(mask)

        output = self.pipe(
            original_image=img,
            mask_image=mask,
            num_inference_steps=num_step,
            eta=0.0,
            jump_length=jump_length,
            jump_n_sample=jump_n_sample,
            generator=self.generator,
        )

        inpainted_images = output.images
        return inpainted_images


if __name__ == "__main__":
    system = System()
    system.launch()
