from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class ReferentialRow:
    specialite: str
    pathologie: str
    symptoms: str
    description: str


def load_referential(path: Path) -> list[ReferentialRow]:
    rows: list[ReferentialRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader, None)
        if header is None:
            return rows
        for row in reader:
            if not row:
                continue
            while len(row) < 4:
                row.append("")
            rows.append(
                ReferentialRow(
                    specialite=row[0].strip(),
                    pathologie=row[1].strip(),
                    symptoms=row[2].strip(),
                    description=row[3].strip(),
                )
            )
    return rows


def build_corpus(rows: list[ReferentialRow]) -> list[str]:
    corpus = []
    for row in rows:
        text = " | ".join(
            part
            for part in [row.pathologie, row.symptoms, row.description]
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
    }
    return mapping.get(duration, 1.0)


def weighted_score(similarity: float, intensity: int, duration: str, location: str) -> float:
    intensity_factor = 0.85 + 0.05 * intensity
    duration_weight = duration_factor(duration)
    location_factor = 1.05 if location.strip() else 1.0
    return similarity * intensity_factor * duration_weight * location_factor


def build_user_text(payload: dict) -> str:
    parts = [
        payload.get("description", ""),
        payload.get("context", ""),
        payload.get("location", ""),
        payload.get("duration", ""),
    ]
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
    location = payload.get("location", "")

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
