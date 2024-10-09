import replicate
import requests


class FluxAgent:
    def __init__(self):
        self.name = "FluxBot"

    def generate_image(self, model, prompt, steps, controlnet, image_url):
        input = {
            "width": 1024,
            "height": 1024,
            "guidance": 3.5,
            "prompt": prompt,
            "output_format": "png",
            "output_quality": 100,
            "num_inference_steps": steps
        }
        if controlnet:
            input["control_image"] = image_url
            input["control_type"] = "canny"
            input["guidance_scale"] = 2.5
            input["control_strength"] = 0.5
            input["depth_preprocessor"] = "DepthAnything"
            input["soft_edge_preprocessor"] = "HED"
            input["image_to_image_strength"] = 0
            input["return_preprocessed_image"] = False

        output = replicate.run(
            model,
            input=input
        )
        return output
