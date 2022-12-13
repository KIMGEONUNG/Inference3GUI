#!/usr/bin/env python

from io import BytesIO

import torch

import PIL
import requests
from diffusers import RePaintPipeline, RePaintScheduler
from pycomar.images import show3plt
import gradio as gr
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


class System(object):

    def __init__(self):

        ##############################################################################################
        repo_id = "stabilityai/stable-diffusion-2"
        self.pipe = DiffusionPipeline.from_pretrained(repo_id).to("cuda")
            # repo_id, torch_dtype=torch.float16, revision="fp16").to("cuda")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config)
        ##############################################################################################

        # Construct UI
        with gr.Blocks() as self.demo:
            with gr.Box():
                with gr.Column():
                    with gr.Row():
                        prompt = gr.Textbox(
                            value=
                            "A high tech solarpunk utopia in the Amazon rainforest",
                            label="Text Prompt")
                        prompt_n = gr.Textbox(
                            value=
                            "low quality",
                            label="Negative Prompt")
                        num_samples = gr.Slider(minimum=1.,
                                                maximum=8,
                                                value=2,
                                                label="Num Sample")
            with gr.Box():
                with gr.Column():
                    btn_inpaint = gr.Button("Generate")
                    view_result = gr.Gallery(label="Result",
                                             interactive=False).style(grid=2)
            # PRELOAD

            btn_inpaint.click(self.inference,
                              inputs=[prompt, prompt_n, num_samples],
                              outputs=view_result)

    def launch(self):
        self.demo.launch()

    def inference(self,
                  prompt,
                  prompt_n,
                  num_samples=4):
        guidance_scale = 7.5

        # Inference
        images = self.pipe(prompt=prompt,
                           negative_prompt=prompt_n,
                           guidance_scale=guidance_scale,
                           num_images_per_prompt=num_samples,
                           num_inference_steps=25).images
        return images


if __name__ == "__main__":
    system = System()
    system.launch()
