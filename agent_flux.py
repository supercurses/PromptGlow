import replicate
import requests

class FluxAgent:
    def __init__(self):
        self.name = "FluxBot"

    def generate_image_2(self, prompt):
        url = "https://api.together.xyz/v1/images/generations"

        payload = {
            "prompt": prompt,
            "model": "black-forest-labs/FLUX.1-schnell",
            "steps": 4,
            "n": 1,
            "height": 1024,
            "width": 1024
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "Bearer b70439b4f44017218b2295b996a7f4d202a567243de9ed9fad4b44099b9c346b"
        }
        response = requests.post(url, headers=headers)
        return response.text
    def generate_image(self, model, prompt, steps, controlnet, image_url):
        input = {
            "width": 856,
            "height": 1156,
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
