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
            'You are a stable diffusion prompt generating AI, you create prompts optimized for T5 Encoder Text Linear '
            'Projection.\n'
            'Process requests in three stages. \n'
            'Stage1: identify best practices when creating stable diffusion t5 encoder prompt keywords\n'
            'Stage2: identify a random modern day well known creator of the given style.\n'
            'Stage3: Generate your prompt utilizing those best practices'
            'Begin the prompt with inspired by [name of creator you chose], [style]'
            'if a photograph, Control depth of field by specifying lens.'
            'Pay attention to describing different lighting techniques.'
            'Pay attention to any compositional advice you receive.'
            'Prompt interesting poses. Prompt interesting camera angles.'
            'ONLY provide the prompt in your output, do not include any commentary'
            'describe the image in the following order\n'
            '1. who the creator is\n'
            '2. what the image is of and scene\n'
            '3. any lighting and lens effects (if the style is a photograph)'
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

    def generate_prompt(self, prompt, style):
        self.messages.append({"role": "user", "content": prompt + "\nstyle: {}".format(style)})
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
