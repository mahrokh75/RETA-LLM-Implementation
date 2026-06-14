import torch
import openai
from config import openai_api_key

@torch.no_grad()
def generate_response(model, tokenizer, input_text, **kwargs):
    if model == "chatgpt":
        openai.api_key = openai_api_key
        response_text = chatgpt_response(input_text)
    else:
        inputs = tokenizer([input_text], padding=True, return_tensors='pt')
        new_input_text = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        device = next(iter(model.parameters())).device
        input_ids = inputs['input_ids'].to(device)
        outputs = model.generate(input_ids, **kwargs)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        response_text = output_text[0][len(new_input_text[0]):].strip()
        del input_ids
    return response_text

def chatgpt_response(input_text):
    openai.api_base = "https://openrouter.ai/api/v1"
    completion = openai.ChatCompletion.create(
        model="openrouter/free",
        messages=[
            {"role": "user", "content": input_text}
        ]
    )
    response_text = completion["choices"][0]["message"].get("content") or ""
    return response_text
