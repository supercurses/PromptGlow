# Pre-requisites

1. Replicate (PAID) account to generate Flux images
2. Together AI account to generate prompts via Meta Llama (currently free)

Generate an API token at [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens), copy the token, then set it as an environment variable in your shell:

```
export REPLICATE_API_TOKEN=r8_....
```

Next, [register for a TogetherAI account](https://api.together.xyz/settings/api-keys) to get an API key. 

Once you've registered, set your account's API key to an environment variable named `TOGETHER_API_KEY`:

```
export TOGETHER_API_KEY=xxxxx
```

Note: TogetherAI can be slow to generate prompts. It may be faster to generate prompts via a local instance of LM Studio. 

In app.py:

```
prompt_agent = PromptAgent(local=True)
```


# Installation

```
git clone https://github.com/supercurses/PromptGlow.git
cd PromptGlow
create venv -> activate
pip install -r requirements.txt
```

# SDXL

Requires a local Automatic 1111 installation with API enabled running on http://127.0.0.1:7860.

All parameters can be modified in agent_SDXL.py
