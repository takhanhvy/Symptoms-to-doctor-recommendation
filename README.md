# Orientation vers un médecin selon les symptômes

## Contexte du projet
Ce projet s’inscrit dans le cadre du **Projet Agent Intelligent Sémantique et Génératif)** du cursus *Data Engineering & AI*.  
Il vise à développer une application web intelligente permettant d’orienter un utilisateur vers une spécialité médicale adaptée, à partir d’une description en langage naturel de ses symptômes.

**** Important** :  
Ce système **ne fournit pas de diagnostic médical**. Il s’agit uniquement d’un outil d’orientation indicative.


## Objectifs
- Analyser des symptômes décrits en texte libre
- Comparer ces descriptions à un référentiel médical structuré
- Utiliser une analyse sémantique (SBERT) pour mesurer la similarité
- Recommander les Top 3 spécialités médicales les plus pertinentes
- Détecter les red flags (signaux d’alerte) nécessitant une consultation rapide
- Générer une explication pédagogique via une IA générative (usage limité et contrôlé)


## Architecture globale (Mini-Agent RAG)

1. **Retrieval**
   - Chargement du référentiel médical
   - Sélection des spécialités pertinentes  
2. **Analyse sémantique**
   - Embeddings SBERT (local)
   - Similarité cosinus  
3. **Scoring**
   - Pondération par intensité des symptômes
   - Classement des spécialités  
4. **Red Flags**
   - Détection de signaux d’alerte  
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
├─ notebooks/           # Exploration et construction du référentiel
├─ scripts/             # Scripts utilitaires (build, tests)
├─ tests/               # Tests unitaires
├─ docs/                # Documentation technique
├─ requirements.txt
├─ .env.example
└─ README.md
