# %%
# !pip install transformers pillow

# # %%
# !pip install matplotlib

import os
os.environ["DISABLE_FLASH_ATTN"] = "1"

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
# %matplotlib inline

#%%
from huggingface_hub import snapshot_download
snapshot_download("microsoft/Florence-2-base", local_dir="./florence2-base", local_dir_use_symlinks=False)

#%% 

model_id = 'microsoft/Florence-2-base'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code = True).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True)


# %%
image_path = r"testbilder/CLXQ7779.JPG"
image = Image.open(image_path).convert("RGB")



def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(
        text=prompt,
        images=image,  # ❗ image ist nicht lokal — muss global sein!
        return_tensors="pt"
    )

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

task_prompt = '<CAPTION>'
result = run_example(task_prompt=task_prompt)
print(result)
# %%
