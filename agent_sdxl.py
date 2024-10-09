from PIL import Image
import requests
import io
import base64


class SDXLAgent:
    """ This class is used to generate SDXL images from a local Automatic 1111 instance
    Supports txt2img, img2img and extras API (for upscaling)
    """
    def __init__(self):
        self.api_url = "http://127.0.0.1:7860"

    def download_and_encode_image(self, image_url):
        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            # Encode the image in base64
            encoded_image = base64.b64encode(response.content).decode('utf-8')
            return encoded_image
        else:
            raise Exception("Failed to download image")

    def upscale_image(self, image_path):
        with open(image_path, 'rb') as file:
            image_data = file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        payload = {
              "resize_mode": 0,
              "gfpgan_visibility": 0,
              "codeformer_visibility": 0,
              "codeformer_weight": 0,
              "upscaling_resize": 4,
              "upscaling_crop": True,
              "upscaler_1": "ESRGAN_4x",
              "upscale_first": False,
              "image": encoded_image
        }

        response = requests.post(url=f'{self.api_url}/sdapi/v1/extra-single-image', json=payload)
        r = response.json()
        sd_image = Image.open(io.BytesIO(base64.b64decode(r['image'])))
        sd_image.save('final_upscaled.png')

    def img2img(self, img2img_prompt, counter, image_path, image_url, adetailer):
        if counter == 1:
            encoded_image = self.download_and_encode_image(image_url)
        else:
            with open(image_path, 'rb') as file:
                image_data = file.read()
            encoded_image = base64.b64encode(image_data).decode('utf-8')

        payload = {

                "prompt": img2img_prompt,
                "sampler_name": "DPM++ SDE Karras",
                "steps": 8,
                "seed": -1,
                "cfg_scale": 3,
                "width": 1024,
                "height": 1024,
                "denoising_strength": 0.45,
                "init_images": [
                    encoded_image
                ],
                "alwayson_scripts": {}

        }

        if adetailer:
            if "alwayson_scripts" not in payload:
                payload["alwayson_scripts"] = {}
            payload["alwayson_scripts"]["ADetailer"] = {
                "args": [
                    {
                        "ad_model": "face_yolov8n.pt",
                    }
                ]
            }

        response = requests.post(url=f'{self.api_url}/sdapi/v1/img2img', json=payload)
        r = response.json()
        sd_image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
        sd_image.save('test_img2img_{}.png'.format(counter))


    def txt2img(self, prompt, id, hires, adetailer, controlnet, image_url):

        payload = {
            "prompt": prompt,
            "steps": 8,
            "cfg_scale": 3,
            "width": 768,
            "height": 1024,
            "sampler_name": "DPM++ SDE Karras",
            "enable_hr": False,

        }

        if hires:
            payload["enable_hr"] = True
            payload["firstphase_width"] = 512
            payload["firstphase_height"] = 512
            payload["hr_scale"] = 2
            payload["hr_upscaler"] = "Latent"
            payload["hr_second_pass_steps"] = 0
            payload["denoising_strength"] = 0.7


        # Conditionally add alwayson_scripts if adetailer is True
        # Conditionally add ADetailer to alwayson_scripts if adetailer is True
        if adetailer:
            if "alwayson_scripts" not in payload:
                payload["alwayson_scripts"] = {}
            payload["alwayson_scripts"]["ADetailer"] = {
                "args": [
                    {
                        "ad_model": "face_yolov8n.pt",
                    }
                ]
            }

        # Conditionally add ControlNet to alwayson_scripts if controlnet is True
        if controlnet:
            if "alwayson_scripts" not in payload:
                payload["alwayson_scripts"] = {}
            payload["alwayson_scripts"]["controlnet"] = {
                "args": [
                    {
                        "is_img2img": False,
                        "is_ui:": False,
                        "enabled": True,
                        "image": self.download_and_encode_image(image_url),
                        "module": "canny",
                        "model": "diffusers_xl_canny_mid [112a778d]",
                        "weight": 0.50,
                        "control_mode": "Balanced"
                    }
                ]
            }

        response = requests.post(url=f'{self.api_url}/sdapi/v1/txt2img', json=payload)
        r = response.json()
        sd_image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
        sd_image.save('output{}.png'.format(id))


def test_img2img():
    test_prompt = (
        "A stunning woman stands alone on a serene lake shore, clad in a flowing white dress "
        "that drapes elegantly from her form-fitting top to her flowing hem. Her long blonde "
        "hair cascades down her back like a golden waterfall as she gazes directly at the viewer "
        "with an enigmatic smile. One arm rests lightly on her hip, while the other hangs loosely "
        "by her side, drawing the eye to her slender fingers and delicate wrists.\n\n"
        "The soft focus of the image captures the serene atmosphere of the lake shore, where the "
        "warm sunlight sets behind a woman in a picturesque scene. The gentle ripples on the water's "
        "surface mirror the subtle movement of her arms, while the surrounding trees and hills fade "
        "softly into the distance. A vibrant orange glow creeps across the sky, casting a warm light "
        "over the tranquil scene.\n\n"
        "In this serene photograph, a beautiful woman stands poised at the water's edge, surrounded "
        "by lush greenery and reflecting the peaceful atmosphere of the lake shore."
    )
    sd = SDXLAgent()
    sd.img2img(test_prompt, "output1.png", adetailer=True)


def test_upscaler():
    sdxl_agent = SDXLAgent()
    sdxl_agent.upscale_image("img2img.png")


