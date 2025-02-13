import argparse
import logging

import torch
from diffusers import StableDiffusionPipeline

from monai.deploy.core import Application


class App(Application):
    name = "Diffusion Image App"
    description = "Simple application showing diffusion to generate Images"

    def compose(self):
        model_id = "Nihirc/Prompt2MedImage"
        device = "cuda"
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_prompt", type=str, default="Generate a X-ray")
        parser.add_argument("--output", type=str, default="./out.jpg")
        args = parser.parse_args()

        input_prompt = args.input_prompt
        output_path = args.output
        print("Input Prompt: ", input_prompt)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        prompt = "Show me an X ray pevic fracture"
        image = pipe(prompt).images[0]
        image.save(output_path)


if __name__ == "__main__":
    logging.info(f"Begin {__name__}")
    App().run()
    logging.info(f"End {__name__}")
