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

        self.system_prompt = (
            "Your task is to provide prompts optimized for generating AI images. Follow these rules:\n\n"
            "- Use clear well-structured language. Avoid overly long convoluted sentences. Stick to natural clear grammar with a focus on the meaning of the sentence.\n"
            "- Use descriptive language but don't overload the prompt with excessive adjectives or details. Make sure every prompt serves to visualize and contextualize the scene.\n"
            "- Avoid adding irrelevant or conflicting details that may distract from the main focus.\n"
            "- Specify the style or medium.\n"
            "- Focus on important scene elements.\n"
            "- Include context or actions that might occur in the scene."
        )

        self.messages.append({"role": "system", "content": self.system_prompt})
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

    def generate_prompt(self, prompt):
        #self.messages.append({"role": "user", "content": prompt + "\nstyle: {}".format(style)})
        self.messages.append({"role": "user", "content": prompt})
        ai_response = self.generate_message(messages=self.messages)
        self.messages.append({"role": "assistant", "content": ai_response.choices[0].message.content})
        self.prompts.append(ai_response.choices[0].message.content)
        return ai_response.choices[0].message.content

    def shrink_prompt(self, prompt):
        self.messages.append({"role": "user", "content": "reduce the number of words in the following prompt while "
                                                         "retaining the meaning. Prompt: {}".format(prompt)})
        ai_response = self.generate_message(messages=self.messages)
        self.messages.append({"role": "assistant", "content": ai_response.choices[0].message.content})
        self.prompts.append(ai_response.choices[0].message.content)
        return ai_response.choices[0].message.content

    def generate_clip_prompt(self, prompt):
        clip_system_prompt = ('Your task is to convert a stable diffusion prompt that has been optimized for t5'
                              'encoding into a prompt that has been optimized for CLIP encoding. Only provide the new '
                              'prompt in your response.\n'
                              'Example:\n'
                              'Grey car park, dog under car, wet fur, barking, rain, smeared wheels, slick pavement')
        clip_prompt = []
        clip_prompt.append({"role": "system", "content": clip_system_prompt})
        clip_prompt.append({"role": "user", "content": prompt})
        ai_response = self.generate_message(messages=clip_prompt)
        return ai_response.choices[0].message.content
