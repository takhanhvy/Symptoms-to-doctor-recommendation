# Orientation vers un médecin selon les symptômes

## Contexte du projet
Ce projet s’inscrit dans le cadre du Projet Agent Intelligent Sémantique et Génératif) du cursus *Data Engineering & AI*.  
Il vise à développer une application web intelligente permettant d’orienter un utilisateur vers une spécialité médicale adaptée, à partir d’une description en langage naturel de ses symptômes.

**Important** :  
Ce système **ne fournit pas de diagnostic médical**. Il s’agit uniquement d’un outil d’orientation indicative.


## Objectifs
- Analyser des symptômes décrits en texte libre
- Comparer ces descriptions à un référentiel médical structuré
- Utiliser une analyse sémantique (SBERT) pour mesurer la similarité
- Recommander les Top 3 spécialités médicales les plus pertinentes
- Générer une explication de style justificatif via une IA générative (usage limité et contrôlé)


## Architecture globale (Mini-Agent RAG)

1. **Retrieval**
   - Chargement du référentiel médical
   - Sélection des spécialités pertinentes  
2. **Analyse sémantique**
   - Embeddings SBERT
   - Similarité cosinus  
5. **Generation (GenAI)**
   - Justification de la recommandation
   - Synthèse utilisateur  
   *(1 appel API par sortie, avec cache)*

---

## Structure du projet

```text
aisca-med-reco/
├─ app/                 # Application Streamlit
├─ src/                 # Code métier (NLP, scoring, GenAI)
├─ data/
│  ├─ raw/              # Datasets sources
│  └─ processed/        # Référentiel médical final
├─ scripts/             # Scripts utilitaires (build, tests)
├─ tests/               # Tests unitaires
├─ docs/                # Documentation technique
├─ requirements.txt
├─ .env.example
└─ README.md
```

---

## Prerequis
- Python 3.10+ recommande
- Un acces internet pour telecharger les modeles SBERT
- Une cle API Gemini (recommandation : Gemini 2.5 flash)

---

## Installation
1. Creer un environnement virtuel
   - Windows (PowerShell):
     `python -m venv venv`
     `venv\\Scripts\\activate`
   - macOS / Linux:
     `python -m venv venv`
     `source venv/bin/activate`
2. Installer les dependances:
   `pip install -r requirements.txt`
3. Configurer la cle API (optionnel pour GenAI):
   - Copier `.env.example` vers `.env` ou `app/.env`
   - Renseigner `GEMINI_API_KEY=...`

---

## Lancer l'application
Depuis la racine du projet:
`streamlit run app/questionnaire.py`

---

## Guide d'utilisation (UI)
1. Decrire vos symptomes avec vos propres mots.
2. Completer les questions complémentaires.
4. Cliquer sur **Analyser**.
5. Consulter:
   - Le Top 3 des specialites recommandees
   - Le radar de comparaison (score final pondere)
   - La justification GenAI (si cle configuree)

---

## Notes importantes
- Ce systeme ne remplace pas un avis medical.
- Si vous avez des symptomes graves ou persistants, consultez un professionnel.
