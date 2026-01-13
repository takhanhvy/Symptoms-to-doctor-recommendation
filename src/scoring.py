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


def duration_factor(duration: str) -> float:
    mapping = {
        "Less than 24 hours": 0.9,
        "1-3 days": 1.0,
        "4-7 days": 1.05,
        "1-4 weeks": 1.1,
        "More than 1 month": 1.15,
        "Moins de 24 heures": 0.9,
        "1-3 jours": 1.0,
        "4-7 jours": 1.05,
        "1-4 semaines": 1.1,
        "Plus d'un mois": 1.15,
    }
    return mapping.get(duration, 1.0)


def weighted_score(similarity: float, intensity: int, duration: str, locations) -> float:
    intensity_factor = 0.85 + 0.05 * intensity
    duration_weight = duration_factor(duration)
    location_items = []
    if isinstance(locations, str):
        location_items = [loc.strip().lower() for loc in locations.split(",") if loc.strip()]
    elif isinstance(locations, list):
        location_items = [str(loc).strip().lower() for loc in locations if str(loc).strip()]

    generic_locations = {"generalized / whole body", "généralisé / tout le corps"}
    specific_locations = [
        loc for loc in location_items if loc and loc not in generic_locations
    ]
    if not specific_locations:
        location_factor = 1.0
    else:
        location_factor = min(1.1, 1.05 + 0.02 * max(0, len(specific_locations) - 1))
    return similarity * intensity_factor * duration_weight * location_factor


def location_relevance_factor(row: ReferentialRow, locations) -> float:
    location_items = []
    if isinstance(locations, str):
        location_items = [loc.strip().lower() for loc in locations.split(",") if loc.strip()]
    elif isinstance(locations, list):
        location_items = [str(loc).strip().lower() for loc in locations if str(loc).strip()]

    generic_locations = {"generalized / whole body", "généralisé / tout le corps"}
    location_items = [loc for loc in location_items if loc and loc not in generic_locations]
    if not location_items:
        return 1.0

    keys = set()
    for loc in location_items:
        if "abdomen" in loc or "belly" in loc or "ventre" in loc:
            keys.add("abdomen")
        elif "back" in loc or loc == "dos":
            keys.add("back")
        elif "chest" in loc or "thorax" in loc or "poitrine" in loc:
            keys.add("chest")
        elif "skin" in loc or "peau" in loc:
            keys.add("skin")

    if not keys:
        return 1.0

    def matches_keyword(text: str, kw: str) -> bool:
        if not kw:
            return False
        if len(kw) <= 3 and kw.isalpha():
            return re.search(rf"\b{re.escape(kw)}\b", text) is not None
        return kw in text

    keywords = {
        "abdomen": {
            "abdomen",
            "abdominal",
            "belly",
            "stomach",
            "ventre",
            "estomac",
            "intestin",
            "gastro",
            "naus",
            "vom",
            "diarr",
            "rectum",
            "anus",
            "anal",
            "urine",
            "urinaire",
            "miction",
            "uriner",
            "vessie",
            "pelv",
            "bassin",
        },
        "back": {"back", "dos", "dors", "lomb", "cerv", "colonne", "spine", "vertèbr", "vertebr", "rein", "reins", "kidney"},
        "chest": {"chest", "thorax", "poitrine", "thorac", "heart", "card", "poumon", "lung"},
        "skin": {"skin", "peau", "cutan", "rash", "éruption", "eruption", "démange", "itch"},
    }

    key_weights = {
        "abdomen": 1.3,
        "back": 1.0,
        "chest": 1.0,
        "skin": 1.0,
    }

    row_text = " ".join([row.pathologie, row.symptoms, row.description]).lower()
    total_weight = sum(key_weights.get(key, 1.0) for key in keys)
    matched_weight = 0.0
    for key in keys:
        if any(matches_keyword(row_text, kw) for kw in keywords.get(key, set())):
            matched_weight += key_weights.get(key, 1.0)
    ratio = (matched_weight / total_weight) if total_weight else 1.0
    return 0.55 + 0.6 * ratio


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
    intensity = int(payload.get("intensity", 3))
    duration = payload.get("duration", "")
    location = payload.get("location", [])

    raw_similarities = similarities.copy()

    location_factors = np.array(
        [location_relevance_factor(row, location) for row in referential], dtype=float
    )
    similarities = similarities * location_factors

    # Debug: Afficher les similarités et les textes du corpus
    top_idx = np.argsort(similarities)[::-1][:5]
    print("\n--- DEBUG SIMILARITES ---")
    print(f"Texte utilisateur: {user_text}")
    for idx in top_idx:
        print(
            f"[{idx}] sim={similarities[idx]:.4f} (raw={raw_similarities[idx]:.4f} loc={location_factors[idx]:.2f}) : {corpus[idx]}"
        )
    print("------------------------\n")

    scored = []
    for row, sim in zip(referential, similarities):
        score = weighted_score(float(sim), intensity, duration, location)
        scored.append(
            {
                "specialite": row.specialite,
                "pathologie": row.pathologie,
                "similarity": float(sim),
                "score": float(score),
                "explanation": (
                    "Proximity between your description and the reference symptoms. "
                    f"Score weighted by intensity ({intensity}), duration, and location."
                ),
            }
        )

    best_by_specialty: dict[str, dict] = {}
    for item in scored:
        key = item["specialite"]
        if key not in best_by_specialty or item["score"] > best_by_specialty[key]["score"]:
            best_by_specialty[key] = item

    results = sorted(best_by_specialty.values(), key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def build_augmented_prompt(user_payload: dict, matched_results: list[dict], lang: str = "en") -> str:
    """
    Augmented Context Construction: build a structured prompt for the LLM
    based on user symptoms and retrieved pathologies.
    """
    user_desc = user_payload.get("description", "")
    intensity = user_payload.get("intensity", 3)
    duration = user_payload.get("duration", "")
    locations = user_payload.get("location", [])
    medical_history = user_payload.get("medical_history", [])
    trigger = user_payload.get("trigger_factor", "")

    locations_str = ", ".join(locations) if isinstance(locations, list) else str(locations)
    history_str = ", ".join(medical_history) if isinstance(medical_history, list) else str(medical_history)

    pathologies_context = "\n".join(
        f"- {r['pathologie']} ({r['specialite']}) – score: {r['score']:.2f}"
        for r in matched_results
    )

    if lang == "fr":
        prompt = f"""Tu es un assistant médical. Analyse les symptômes de l'utilisateur et les pathologies identifiées par le système RAG.

**Symptômes décrits par l'utilisateur:**
{user_desc}

**Informations complémentaires:**
- Intensité: {intensity}/5
- Durée: {duration}
- Localisation: {locations_str}
- Antécédents médicaux: {history_str}
- Facteur déclenchant: {trigger}

**Pathologies identifiées par similarité sémantique (top matches):**
{pathologies_context}

**Instructions:**
1. Explique brièvement pourquoi ces pathologies ont été identifiées comme pertinentes.
2. Indique quel spécialiste consulter en priorité.
3. Donne des conseils généraux (sans remplacer un avis médical).
4. Précise les signes d'alerte nécessitant une consultation urgente.

Réponds de manière concise et professionnelle en français."""
    else:
        prompt = f"""You are a medical assistant. Analyze the user's symptoms and the pathologies identified by the RAG system.

**User-described symptoms:**
{user_desc}

**Additional information:**
- Intensity: {intensity}/5
- Duration: {duration}
- Location: {locations_str}
- Medical history: {history_str}
- Trigger factor: {trigger}

**Pathologies identified by semantic similarity (top matches):**
{pathologies_context}

**Instructions:**
1. Briefly explain why these pathologies were identified as relevant.
2. Indicate which specialist to consult first.
3. Provide general advice (without replacing medical opinion).
4. Specify warning signs requiring urgent consultation.

Respond concisely and professionally in English."""

    return prompt


def generate_gemini_response(prompt: str, model_name: str | None = None, temperature: float = 0.3, max_output_tokens: int = 512, max_retries: int = 3) -> str:
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