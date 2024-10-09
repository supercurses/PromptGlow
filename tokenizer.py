from transformers import CLIPTokenizer

# Load the tokenizer used by the Stable Diffusion model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")


class Tokenizer:
    def __init__(self):
        self.name = "Tokenizer"

    def get_sequence_length(self, prompt):
        tokens = tokenizer.tokenize(prompt)
        return len(tokens)

tk = Tokenizer()
print(tk.get_sequence_length("Hello you, I am you"))