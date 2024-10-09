import os
from together import Together
import base64
import io
from PIL import Image

class ReviewAgent:
    def __init__(self):
        self.client = Together(api_key=os.environ.get('TOGETHER_AI_KEY'))

    def halve_image_size(self, image):
        # Get the current size of the image
        width, height = image.size
        # Halve the size
        new_size = (width // 2, height // 2)
        # Resize the image
        resized_image = image.resize(new_size)
        return resized_image

    def image_to_base64(self, image):
        # Convert the PIL image to bytes and then base64 encode it
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str


    def check_text(self, text):
        message = self.client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            max_tokens=1024,
            temperature=0.3,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "The following text has scanned from a page by an AI "
                                    "determine whether the text contains nonsensical words. If there are no problems "
                                    "begin your response with !@! : {}".format(text)

                        }

                    ],
                }

            ],
        )
        return message.choices[0].message.content

    def review_image(self, url):
        #image = Image.open(image_path)
        # Halve the image size before encoding it
        #resized_image = self.halve_image_size(image)
        # Convert the resized image to base64
        #image_base64 = self.image_to_base64(image)
        message = self.client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            max_tokens=1024,
            temperature=0.1,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=False,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a professional stable diffusion artist. "
                                    "make suggestions to improve the attached image. "
                                    "Consider the composition, pose,"
                                    "lighting, colour, contrast, scene."

                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": url
                            }
                        },
                        {
                            "type": "text",
                            "text": "here is the image"


                        }
                    ]
                },



            ],
        )
        return message.choices[0].message.content
