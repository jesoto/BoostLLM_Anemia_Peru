"""Inferencia MedGemma: P(Sí) desde logits de tokens, nunca confianza verbalizada.

Requiere `pip install -e .[llm]` y acceso a los pesos (licencia HAI-DEF en Hugging Face).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class MedGemmaScorer:
    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it", dtype=torch.bfloat16):
        # Nota: la variante 4B es multimodal; si AutoModelForCausalLM no la carga, usar la clase indicada en su model card.
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto").eval()
        self.yes = self.tok.encode("Sí", add_special_tokens=False)[0]
        self.no = self.tok.encode("No", add_special_tokens=False)[0]

    @torch.no_grad()
    def prob_yes(self, prompt: str) -> float:
        msgs = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        logits = self.model(ids).logits[0, -1]
        pair = torch.stack([logits[self.yes], logits[self.no]]).float()
        return torch.softmax(pair, dim=0)[0].item()  # normalizado sobre {Sí, No}


def route_by_uncertainty(p_xgb, p_llm, band=(0.35, 0.65)):
    """Usa el LLM solo si XGBoost cae en la zona de incertidumbre; si no, conserva XGBoost."""
    lo, hi = band
    return [pl if lo <= px <= hi else px for px, pl in zip(p_xgb, p_llm)]
