#!/usr/bin/env python

from io import BytesIO

import torch

import PIL
import requests
from diffusers import RePaintPipeline, RePaintScheduler
from pycomar.images import show3plt
import gradio as gr
import numpy as np
from diffusers import (
    StableDiffusionInpaintPipeline, )


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


class System(object):

    def __init__(self):

        ckpt = "runwayml/stable-diffusion-inpainting"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(ckpt).to(
            "cuda")
        self.generator = torch.Generator(device="cuda").manual_seed(0)

        # Construct UI
        img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
        original_image = download_image(img_url).resize((512, 512))
        mask_image = download_image(mask_url).resize((512, 512))
        with gr.Blocks() as self.demo:
            with gr.Box():
                with gr.Column():
                    btn_get_mask = gr.Button("Get Mask")
                    with gr.Row():
                        mask = gr.ImageMask(label="Inpaining")
                        view_original = gr.Image(label="Original",
                                                 interactive=False)
                        view_mask = gr.Image(label="Mask", interactive=False)
                    with gr.Row():
                        prompt = gr.Textbox(
                            value="a mecha robot sitting on a bench",
                            label="Text Prompt")
                        prompt_n = gr.Textbox(
                            value=
                            "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, artifact, noised",
                            label="Negative Prompt")
                        strength = gr.Slider(minimum=0.,
                                             maximum=1.,
                                             value=0.75,
                                             label="Strength")
                        num_samples = gr.Slider(minimum=1.,
                                                maximum=16,
                                                value=4,
                                                label="Num Sample")
            with gr.Box():
                with gr.Column():
                    btn_inpaint = gr.Button("Inpaint")
                    view_result = gr.Gallery(label="Result",
                                             interactive=False).style(grid=4)
            # PRELOAD
            self.demo.load(lambda: original_image, outputs=mask)
            self.demo.load(lambda: original_image, outputs=view_original)
            self.demo.load(lambda: mask_image, outputs=view_mask)

            btn_get_mask.click(lambda x: x["image"],
                               inputs=mask,
                               outputs=view_original)
            btn_get_mask.click(lambda x: x["mask"][..., :3],
                               inputs=mask,
                               outputs=view_mask)
            btn_inpaint.click(self.inference,
                              inputs=[
                                  view_original, view_mask, prompt, prompt_n,
                                  strength, num_samples
                              ],
                              outputs=view_result)

    def launch(self):
        self.demo.launch()

    def inference(self,
                  img: np.ndarray,
                  mask: np.ndarray,
                  prompt,
                  prompt_n,
                  strength,
                  num_samples=4):
        img = PIL.Image.fromarray(img)
        mask = PIL.Image.fromarray(mask)

        # Inference
        guidance_scale = 7.5

        images = self.pipe(
            prompt=prompt,
            negative_prompt=prompt_n,
            image=img,
            mask_image=mask,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=self.generator,
            num_images_per_prompt=num_samples,
        ).images

        return images


if __name__ == "__main__":
    system = System()
    system.launch()
