"""
florence2_module.py
~~~~~~~~~~~~~~~~~~~
Wrapper um Microsoft Florence-2 für Caption- und (rudimentäres) VQA-Inference.
"""

from __future__ import annotations
import os, re
from typing import Literal

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class Florence2Classifier:
    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        device: str | None = None,
        disable_flash_attn: bool = True,
    ):
        if disable_flash_attn:
            os.environ.setdefault("DISABLE_FLASH_ATTN", "1")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id,
                                                       trust_remote_code=True)
        self.model = (AutoModelForCausalLM
                      .from_pretrained(model_id, trust_remote_code=True)
                      .to(self.device)
                      .eval())

    # ────────────────────────────── API ─────────────────────────────
    def classify(
        self,
        image: Image.Image,
        task_prompt: Literal["<CAPTION>", "<VQA>"] = "<CAPTION>",
        text_input: str | None = None,
    ) -> str:
        # -------- Prompt bauen --------------------------------------
        if task_prompt == "<CAPTION>":
            prompt = "<CAPTION>"
        elif task_prompt == "<VQA>":
            if text_input is None:
                raise ValueError("Für <VQA> muss text_input gesetzt sein.")
            prompt = f"<VQA>Question:{text_input.rstrip('?')}? Answer:"
        else:
            raise ValueError("Unbekannter task_prompt")

        # -------- Inference -----------------------------------------
        inputs = self.processor(text=prompt,
                                images=image,
                                return_tensors="pt"
                               ).to(self.device, dtype=self.model.dtype)

        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                num_beams=3,
            )

        raw = self.processor.batch_decode(gen_ids,
                                          skip_special_tokens=True)[0]

        if task_prompt == "<VQA>":
            ans = raw.split("Answer:")[-1].strip()
            return re.sub(r"<loc_\d+>", "", ans).strip()

        # CAPTION
        return raw.strip()
