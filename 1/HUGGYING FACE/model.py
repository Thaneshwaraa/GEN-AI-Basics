import os

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Never hardcode tokens. Set HF_TOKEN in your environment (or a .env file, which is gitignored).
# distilgpt2 is a public model, so a token is optional.
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

model_id = "distilgpt2"
print("Loading model...")  # To check the model is loaded or not

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


def chat(prompt: str) -> str:
    response = pipe(prompt,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,  # must be between 0 and 1
                    return_full_text=False)
    return response[0]["generated_text"]
