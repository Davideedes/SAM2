# florence2_module.py
"""
Wrapper um Microsoft Florence 2 – jetzt mit robuster Pothole‑Detection.
• <CAPTION>, <DETAILED_CAPTION>
• <OPEN_VOCABULARY_DETECTION>
• detect_pothole():  Synonym‑Liste, Score‑Fallback = 1.0, Debug‑Logging
"""
from __future__ import annotations
import os, json
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
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM
            .from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
            .to(self.device)
            .eval()
        )

    # ────────────────────────────── Generische Inference ──────────────────────────────
    def classify(
        self,
        image: Image.Image,
        task_prompt: Literal[
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<OPEN_VOCABULARY_DETECTION>",
        ] = "<CAPTION>",
        text_input: str | None = None,
        max_tokens: int = 128,
    ):
        if task_prompt in ("<CAPTION>", "<DETAILED_CAPTION>"):
            prompt = task_prompt
        elif task_prompt == "<OPEN_VOCABULARY_DETECTION>":
            if text_input is None:
                raise ValueError("Für <OPEN_VOCABULARY_DETECTION> text_input angeben!")
            prompt = task_prompt + text_input
        else:
            raise ValueError(f"Unbekannter task_prompt: {task_prompt}")

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, dtype=self.model.dtype
        )
        with torch.inference_mode():
            gen_ids = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False, num_beams=3
            )
        raw_txt = self.processor.batch_decode(gen_ids, skip_special_tokens=False)[0]

        if task_prompt == "<OPEN_VOCABULARY_DETECTION>":
            det = self.processor.post_process_generation(
                raw_txt,
                task=task_prompt,
                image_size=(image.width, image.height),
            )["<OPEN_VOCABULARY_DETECTION>"]
            return det  # dict mit bboxes / labels / scores
        return raw_txt.strip()

    # ────────────────────────────── Pothole‑Detection ──────────────────────────────
    def detect_pothole(
        self,
        image: Image.Image,
        synonyms: str | None = None,
        score_thr: float = 0.45,
        debug: bool = False,
    ) -> tuple[bool, list[tuple[list[int], float]]]:
        """Liefert (has_pothole, [(bbox, score), …]).
        * Fehlen Scores, wird 1.0 angenommen (Zero‑Shot‑Fallback).
        * Nur Labels, die zu den Synonymen passen, werden gewertet.
        """
        query = synonyms or (
            "pothole, road pothole, road hole, hole in the road, sinkhole, damaged asphalt"
        )
        syn_list = [s.strip().lower() for s in query.split(",")]

        det = self.classify(
            image=image,
            task_prompt="<OPEN_VOCABULARY_DETECTION>",
            text_input=query,
            max_tokens=256,
        )
        if debug:
            print("[Florence‑DEBUG] raw detection\n" + json.dumps(det, indent=2))

        bboxes = det.get("bboxes", [])
        labels = [lbl.lower() for lbl in det.get("labels", [])]
        scores_raw = det.get("scores")
        scores = scores_raw if scores_raw else [1.0] * len(bboxes)

        results, has_pothole = [], False
        for bbox, lbl, scr in zip(bboxes, labels, scores):
            if any(syn in lbl for syn in syn_list):
                results.append((bbox, scr))
                has_pothole |= scr >= score_thr
        return has_pothole, results
