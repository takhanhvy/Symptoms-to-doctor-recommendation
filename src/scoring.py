from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import google.genai as genai
import os

@dataclass(frozen=True)
class ReferentialRow:
    specialite: str
    pathologie: str
    symptoms: str
    description: str


def load_referential(path: Path) -> list[ReferentialRow]:
    rows: list[ReferentialRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        # Support both EN and FR column names
        colmap = {k.lower(): k for k in reader.fieldnames}
        col_specialite = colmap.get("speciality") or colmap.get("specialite")
        col_pathologie = colmap.get("disease") or colmap.get("maladie")
        col_symptoms = colmap.get("symptoms") or colmap.get("symptomes")
        col_description = colmap.get("description")
        for row in reader:
            rows.append(
                ReferentialRow(
                    specialite=row.get(col_specialite, "").strip(),
                    pathologie=row.get(col_pathologie, "").strip(),
                    symptoms=row.get(col_symptoms, "").strip(),
                    description=row.get(col_description, "").strip(),
                )
            )
    return rows


def build_corpus(rows: list[ReferentialRow]) -> list[str]:
    corpus = []
    for row in rows:
        symptoms_text = row.symptoms or row.description
        text = " | ".join(
            part
            for part in [row.pathologie, symptoms_text]
            if part
        )
        corpus.append(text)
    return corpus


@lru_cache(maxsize=1)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def embed_referential(model_name: str, corpus_key: str, corpus: tuple[str, ...]):
    model = load_model(model_name)
    embeddings = model.encode(list(corpus), normalize_embeddings=True)
    return embeddings




def location_relevance_factor(row: ReferentialRow, locations) -> float:
    return 1.0


def build_user_text(payload: dict) -> str:
    parts = [
        payload.get("description", ""),
    ]
    trigger = payload.get("trigger_factor", "")
    trigger_norm = str(trigger).strip().lower()
    if trigger_norm and trigger_norm not in {"none of the options", "aucune des options"}:
        parts.append(str(trigger).strip())
    trigger_details = payload.get("trigger_details", "")
    if trigger_details:
        parts.append(trigger_details)
    locations = payload.get("location", [])
    generic_locations_norm = {"generalized / whole body", "généralisé / tout le corps"}
    if isinstance(locations, list):
        cleaned_locations = [
            str(loc).strip()
            for loc in locations
            if str(loc).strip() and str(loc).strip().lower() not in generic_locations_norm
        ]
        if cleaned_locations:
            parts.append(" ".join(cleaned_locations))
    else:
        location_str = str(locations).strip()
        if location_str and location_str.lower() not in generic_locations_norm:
            parts.append(location_str)
    return " | ".join(part for part in parts if part)


def score_user(
    payload: dict,
    referential: list[ReferentialRow],
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 3,
) -> list[dict]:
    user_text = build_user_text(payload)
    if not user_text:
        return []

    corpus = build_corpus(referential)
    embeddings = embed_referential(model_name, str(hash(tuple(corpus))), tuple(corpus))
    model = load_model(model_name)
    user_embedding = model.encode([user_text], normalize_embeddings=True)[0]

    similarities = np.dot(embeddings, user_embedding)

    raw_similarities = similarities.copy()

    # Debug: Afficher les similarités et les textes du corpus
    top_idx = np.argsort(similarities)[::-1][:5]
    print("\n--- DEBUG SIMILARITES ---")
    print(f"Texte utilisateur: {user_text}")
    for idx in top_idx:
        print(
            f"[{idx}] sim={similarities[idx]:.4f} (raw={raw_similarities[idx]:.4f}) : {corpus[idx]}"
        )
    print("------------------------\n")

    # Calculer scored AVANT le debug Gemini
    scored = []
    for row, sim in zip(referential, similarities):
        scored.append(
            {
                "specialite": row.specialite,
                "pathologie": row.pathologie,
                "similarity": float(sim),
            }
        )

    # DEBUG : Afficher l'analyse générée par Gemini pour ce payload
    try:
        from .scoring import build_augmented_prompt, generate_gemini_response
    except ImportError:
        # fallback si import relatif échoue (exécution directe)
        from src.scoring import build_augmented_prompt, generate_gemini_response
    try:
        lang = "fr" if payload.get("lang") == "fr" else "en"
        prompt = build_augmented_prompt(payload, scored[:top_k], lang=lang)
        print("\n--- DEBUG GEMINI ANALYSE ---")
        print("Prompt envoyé à Gemini :\n", prompt)
        gemini_text = generate_gemini_response(prompt)
        print("Réponse Gemini :\n", gemini_text)
        print("-----------------------------\n")
    except Exception as e:
        print(f"[DEBUG] Erreur lors de l'appel Gemini : {e}")

    best_by_specialty: dict[str, dict] = {}
    for item in scored:
        key = item["specialite"]
        if key not in best_by_specialty or item["similarity"] > best_by_specialty[key]["similarity"]:
            best_by_specialty[key] = item

    results = sorted(best_by_specialty.values(), key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]



def build_augmented_prompt(user_payload: dict, matched_results: list[dict], lang: str = "en") -> str:
    """
    Augmented Context Construction: build a structured prompt for the LLM
    based on user symptoms and retrieved pathologies.
    """
    user_desc = user_payload.get("description", "")
    pathologies_context = "\n".join(
        f"- {r['pathologie']} ({r['specialite']}) – similarity: {r['similarity']:.2f}"
        for r in matched_results
    )

    if lang == "fr":
        prompt = f"""Génère une explication d'orientation médicale concise et professionnelle (3-4 phrases). Inclure : raisonnement, niveau d'urgence, prochaines étapes.

Tu es un assistant médical. Analyse les symptômes décrits et les meilleures correspondances du système.

**Symptômes décrits par l'utilisateur :**
{user_desc}

**Pathologies identifiées par similarité sémantique (top matches) :**
{pathologies_context}

Réponds de manière concise et professionnelle en français."""
    else:
        prompt = f"""Generate a concise, professional medical orientation explanation (3-4 sentences). Include: reasoning, urgency level, and next steps.

You are a medical assistant. Analyze the user's described symptoms and the top similarity matches.

**User-described symptoms:**
{user_desc}

**Pathologies identified by semantic similarity (top matches):**
{pathologies_context}

Respond concisely and professionally in English."""

    return prompt


def generate_gemini_response(prompt: str, model_name: str | None = None, temperature: float = 0.3, max_output_tokens: int = 4096, max_retries: int = 3) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    model_name = model_name or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"

    client = genai.Client(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                },
            )
            return response.text if getattr(response, "text", None) else str(response)
        except Exception as e:
            err_text = str(e)
            is_rate_limit = "429" in err_text or "RESOURCE_EXHAUSTED" in err_text
            is_overloaded = (
                "503" in err_text or "UNAVAILABLE" in err_text or "overloaded" in err_text.lower()
            )

            if is_rate_limit and "limit: 0" in err_text:
                raise

            if (is_rate_limit or is_overloaded) and attempt < max_retries - 1:
                retry_delay = None
                match = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", err_text)
                if match:
                    retry_delay = float(match.group(1))
                else:
                    match = re.search(r"retryDelay['\"]:\s*['\"]([0-9]+)s", err_text)
                    if match:
                        retry_delay = float(match.group(1))

                wait_time = (2 ** attempt) * (10 if is_rate_limit else 5)
                if retry_delay is not None:
                    wait_time = max(wait_time, retry_delay)

                reason = "Rate limit" if is_rate_limit else "Model overloaded"
                print(
                    f"{reason}, waiting {int(wait_time)}s before retry {attempt + 1}/{max_retries}..."
                )
                time.sleep(wait_time)
                continue

            raise