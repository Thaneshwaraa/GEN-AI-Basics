from huggingface_hub import login
login("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")#login to huggingface hub with an token

from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline

model_id="distilgpt2"
print("Loading model...") #To check the model is loaded or not

tokenizer= AutoTokenizer.from_pretrained(model_id)
model= AutoModelForCausalLM.from_pretrained (model_id)
pipe=pipeline("text-generation",model=model,tokenizer=tokenizer)


def chat(prompt: str) -> str:
    response = pipe(prompt,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=50,
                    return_full_text=False)
    return response[0]["generated_text"]



