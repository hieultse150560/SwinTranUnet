import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("upstage/llama-30b-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "upstage/llama-30b-instruct",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
)

prompt = '''### User:\nYour task is to create a simple description of an image.
The description must be simple, in one sentence length. The description should contain 1-3 objects separately placed on the container. Objects must be graspable, do not reuse generated objects in previously generated descriptions. The description only shows the name of objects, don't describe the objects too much.
You can also use synonyms of objects to increase diversity, for example, mug, cup, and bottle can be considered synonymous.
You must return only the description, nothing else. Don't include any explanation.

\n\n### Assistant:\n'''
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
del inputs["token_type_ids"]
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

output = model.generate(**inputs, streamer=streamer, use_cache=True, max_new_tokens=float('inf'))
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
