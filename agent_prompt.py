import os
from openai import OpenAI
from together import Together


class PromptAgent:
    """A class to ask an LLM to provide Stable Diffusion prompts based on guidance from the user
    Llama 3.2 is a little unreliable at randomness, so we use a genders and ethnicities list to create some variation"""

    def __init__(self, local):
        self.local = local
        if self.local:
            self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            self.model = "lmstudio-community/Llama-3.2-3B-Instruct-GGUF"
        else:
            self.client = Together(api_key=os.environ.get('TOGETHER_AI_KEY'))
            self.model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"

        self.messages = []

        self.t5_system_prompt = (
            "Your task is to provide prompts optimized for generating AI images.\n\n"
            "Process in two stages:\n\n"
            "1. If a type of art and media is provided, identify a well-known expert of that type of art/media in your prompt.\n"
            "Example input:\n"
            "type of art: photograph\n"
            "media: digital\n"
            "prompt: girl riding a bicycle\n"
            "expected output\n"
            "photograph in the style of Erwin Olaf of a young girl riding a bicycle down a dirt path surrounded by tall trees and wildflowers, with a soft focus effect to capture the natural beauty of the scene\n\n"
            "2. Generate the prompt using these rules:\n"
            "- Use clear, well-structured language. Avoid overly long convoluted sentences. Stick to natural clear grammar with a focus on the meaning of the sentence.\n"
            "- Use descriptive language but don't overload the prompt with excessive adjectives or details. Make sure every prompt serves to visualize and contextualize the scene.\n"
            "- Avoid adding irrelevant or conflicting details that may distract from the main focus.\n"
            "- Specify the style or medium.\n"
            "- Focus on important scene elements.\n"
            "- Include context or actions that might occur in the scene.\n"
            "- Only include the prompt in your response, do not include any commentary or reason\n"
            "- Do not introduce the prompt, for example, do not say 'here is the revised prompt'"
        )
        self.clip_system_prompt = ('Your task is to convert a stable diffusion prompt that has been optimized for t5'
                                   'encoding into a prompt that has been optimized for CLIP encoding. Only provide '
                                   'the new prompt in your response.\n'
                                   'Example:\n'
                                   'Grey car park, dog under car, wet fur, barking, rain, smeared wheels, '
                                   'slick pavement')
        self.shrink_system_prompt = 'Reduce the number of words in the provided prompt while retaining the meaning'
        self.prompts = []

    def generate_message(self, messages):
        """ Attempt to get a response from the AI API"""
        try:
            if not self.local:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.7,
                    top_k=50,
                    repetition_penalty=1,
                    stop=["<|eot_id|>", "<|eom_id|>"],
                    truncate=130560,
                    stream=False
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
            return response
        except Exception as e:
            return {"error": str(e)}

    def generate_prompt(self, art_type, media, prompt):
        message = []
        message.append({"role": "system", "content": self.t5_system_prompt})
        message.append({"role": "user", "content": "Art Type: " + art_type + "\nMedia:" + media + "\nPrompt:" + prompt})
        ai_response = self.generate_message(messages=message)
        self.prompts.append(ai_response.choices[0].message.content)
        return ai_response.choices[0].message.content

    def shrink_prompt(self, prompt):
        message = []
        message.append({"role": "system", "content": self.shrink_system_prompt})
        message.append({"role": "user", "content": prompt})
        ai_response = self.generate_message(messages=message)
        self.prompts.append(ai_response.choices[0].message.content)
        return ai_response.choices[0].message.content

    def generate_clip_prompt(self, prompt):
        message = []
        message.append({"role": "system", "content": self.clip_system_prompt})
        message.append({"role": "user", "content": prompt})
        ai_response = self.generate_message(messages=message)
        return ai_response.choices[0].message.content
