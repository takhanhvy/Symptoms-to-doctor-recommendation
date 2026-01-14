import json
import hashlib
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Load .env file from app/ directory
load_dotenv(ROOT / "app" / ".env", override=True)

from src.scoring import load_referential, score_user, build_augmented_prompt, generate_gemini_response

def main():
    st.set_page_config(page_title="Medical Orientation Questionnaire")
    st.title("Medical Orientation Questionnaire")

    # Sélecteur de langue
    lang = st.selectbox("Language / Langue", ["English", "Français"], index=0)
    disclaimer_text = (
        "**Avertissement**: Cet outil est fourni à titre informatif uniquement. Il ne remplace pas un diagnostic médical professionnel. En cas de symptômes sévères ou persistants, consultez un médecin sans délai."
        if lang == "Français"
        else "**Warning**: This tool is for informational purposes only. It is not a substitute for professional medical diagnosis. If you experience severe or persistent symptoms, consult a doctor immediately."
    )
    st.warning(disclaimer_text)
    if lang == "Français":
        referential_path = Path("data/processed/medical_referential_fr.csv")
    else:
        referential_path = Path("data/processed/medical_referential.csv")
    referential = load_referential(referential_path)

    def required_label(text):
        st.markdown(f"{text} <span style='color:red'>*</span>", unsafe_allow_html=True)

    # Dictionnaire de traduction
    translations = {
        "en": {
            "describe": "Describe your situation in your own words",
            "describe_ph": "Example: headache with nausea since yesterday...",
            "location": "Location / body area (select all that apply)",
            "location_opts": [
                "Head / Skull", "Eyes", "Ears", "Nose / Throat", "Chest / Thorax", "Abdomen / Belly", "Back", "Arms / Hands", "Legs / Feet", "Skin", "Generalized / Whole Body", "Other"
            ],
            "specify_other": "Specify other location",
            "specify_other_ph": "Example: neck, jaw, pelvis...",
            "history": "Do you have any of the following medical conditions or risk factors? (multiple choices possible)",
            "history_opts": [
                "Heart problems / Hypertension", "Diabetes", "Respiratory problems / Asthma", "Known allergies", "Chronic digestive problems", "Autoimmune diseases", "Cancer history", "Neurological disorders", "Kidney problems", "Recent surgical operations", "Current medication treatment", "No particular medical history"
            ],
            "specify_condition": "Specify your condition",
            "specify_condition_ph": "Example: type 2 diabetes, seasonal asthma...",
            "trigger": "When do symptoms appear or worsen?",
            "trigger_opts": [
                "None of the options", "During physical effort / exercise", "At rest / during night", "After meals", "When lying down / standing up", "During stressful situations", "Upon contact with certain substances (food, allergens...)", "In cold / hot weather"
            ],
            "trigger_detail": "Specify if trigger exists",
            "trigger_detail_ph": "Example: after long screen time, during travel...",
            "analyze": "Analyze",
            "response_captured": "Response captured.",
            "top_reco": "Top recommendations",
            "computing": "Computing similarities...",
            "please_complete": "Please complete:",
            "no_history_info": "'No particular medical history' was chosen.",
            "table_specialty": "Specialty",
            "table_disease": "Best-matching disease",
            "table_similarity": "Similarity",
            "table_score": "Score",
            "longer_desc": "Please provide a longer description to compute recommendations.",
            "ai_analysis": "AI-Powered Analysis",
            "generating": "Generating personalized recommendations...",
            "gemini_error": "Could not generate AI response. Please check your API key, quota, or try again later (service overloaded).",
            "footer": "**EFREI Project - Generative AI & Semantic Analysis 2025**  \nRAG System with SBERT + Gemini 2.5 Flash | Medical Orientation Prototype| \nMADOUNGOU Colombe Alice, Vy Khanh TA"
        },
        "fr": {
            "describe": "Décrivez votre situation avec vos propres mots",
            "describe_ph": "Exemple : mal de tête avec nausées depuis hier...",
            "intensity": "Intensité de la douleur/gêne (1 = faible, 5 = très forte)",
            "duration": "Durée",
            "duration_opts": [
                "Moins de 24 heures",
                "1-3 jours",
                "4-7 jours",
                "1-4 semaines",
                "Plus d'un mois",
            ],
            "location": "Localisation / zone du corps (sélection multiple possible)",
            "location_opts": [
                "Tête / Crâne", "Yeux", "Oreilles", "Nez / Gorge", "Poitrine / Thorax", "Abdomen / Ventre", "Dos", "Bras / Mains", "Jambes / Pieds", "Peau", "Généralisé / Tout le corps", "Autre"
            ],
            "specify_other": "Précisez une autre localisation",
            "specify_other_ph": "Exemple : cou, mâchoire, bassin...",
            "history": "Avez-vous l'une des conditions médicales ou facteurs de risque suivants ? (choix multiples possibles)",
            "history_opts": [
                "Problèmes cardiaques / Hypertension", "Diabète", "Problèmes respiratoires / Asthme", "Allergies connues", "Problèmes digestifs chroniques", "Maladies auto-immunes", "Antécédents de cancer", "Troubles neurologiques", "Problèmes rénaux", "Opérations chirurgicales récentes", "Traitement médicamenteux en cours", "Aucun antécédent médical particulier"
            ],
            "specify_condition": "Précisez votre condition",
            "specify_condition_ph": "Exemple : diabète de type 2, asthme saisonnier...",
            "trigger": "Quand les symptômes apparaissent-ils ou s'aggravent-ils ?",
            "trigger_opts": [
                "Aucune des options", "Lors d'un effort physique / exercice", "Au repos / pendant la nuit", "Après les repas", "En position allongée / debout", "En situation de stress", "Au contact de certaines substances (aliments, allergènes...)", "Par temps froid / chaud"
            ],
            "trigger_detail": "Précisez si un facteur déclenchant existe",
            "trigger_detail_ph": "Exemple : après un temps d'écran prolongé, lors d'un voyage...",
            "analyze": "Analyser",
            "response_captured": "Réponse enregistrée.",
            "top_reco": "Meilleures recommandations",
            "computing": "Calcul des similarités...",
            "please_complete": "Veuillez compléter :",
            "no_history_info": "'Aucun antécédent médical particulier' a été choisi.",
            "table_specialty": "Spécialité",
            "table_disease": "Symptômes (référentiel)",
            "table_similarity": "Similarité",
            "table_score": "Score",
            "longer_desc": "Merci de fournir une description plus détaillée pour obtenir des recommandations.",
            "ai_analysis": "Analyse par IA",
            "generating": "Génération des recommandations personnalisées...",
            "gemini_error": "Impossible de générer la réponse IA. Vérifiez votre clé API, votre quota, ou réessayez plus tard (service surchargé).",
            "footer": "**Projet EFREI - IA Générative & Analyse Sémantique 2025**  \nSystème RAG avec SBERT + Gemini 2.5 Flash | Prototype d'orientation médicale\nRAG System with SBERT + Gemini 2.5 Flash | Medical Orientation Prototype | \nMADOUNGOU Colombe Alice, Vy Khanh TA"
        }
    }
    t = translations["fr" if lang == "Français" else "en"]

    required_label(t["describe"])
    description = st.text_area(
        t["describe"],
        placeholder=t["describe_ph"],
        label_visibility="collapsed",
    )

    location_options = t["location_opts"]
    required_label(t["location"])
    location_choice = st.multiselect(
        t["location"],
        location_options,
        key=f"location_{lang}",
        label_visibility="collapsed",
    )
    location_other = ""
    other_label = "Other" if lang == "English" else "Autre"
    if other_label in location_choice:
        required_label(t["specify_other"])
        location_other = st.text_input(
            t["specify_other"],
            placeholder=t["specify_other_ph"],
            label_visibility="collapsed",
        )
    location = [loc for loc in location_choice if loc != other_label]
    if location_other.strip():
        location.append(location_other.strip())

    history_options = t["history_opts"]

    def enforce_medical_history():
        key = f"medical_history_{lang}"
        selected = st.session_state.get(key, [])
        none_label = (
            "No particular medical history"
            if lang == "English"
            else "Aucun antécédent médical particulier"
        )
        if none_label in selected and len(selected) > 1:
            st.session_state[key] = [none_label]
            st.session_state[f"medical_history_locked_{lang}"] = True
        else:
            st.session_state[f"medical_history_locked_{lang}"] = False

    required_label(t["history"])
    medical_history = st.multiselect(
        t["history"],
        history_options,
        key=f"medical_history_{lang}",
        on_change=enforce_medical_history,
        label_visibility="collapsed",
    )
    if st.session_state.get(f"medical_history_locked_{lang}"):
        st.info(t["no_history_info"])

    medical_history_details = ""
    none_label = (
        "No particular medical history"
        if lang == "English"
        else "Aucun antécédent médical particulier"
    )
    if medical_history and none_label not in medical_history:
        required_label(t["specify_condition"])
        medical_history_details = st.text_input(
            t["specify_condition"],
            placeholder=t["specify_condition_ph"],
            label_visibility="collapsed",
        )

    # On ne demande plus intensity ni duration
    intensity = None
    duration = None

    # Concatène location + history à la description (entrée utilisateur)
    concat_desc = description.strip()
    if location:
        concat_desc += (
            "\nLocation: " + ", ".join(location)
            if lang == "English"
            else "\nLocalisation : " + ", ".join(location)
        )
    if medical_history:
        concat_desc += (
            "\nMedical history: " + ", ".join(medical_history)
            if lang == "English"
            else "\nAntécédents médicaux : " + ", ".join(medical_history)
        )
    description = concat_desc

    required_label(t["trigger"])
    trigger_factor = st.selectbox(
        t["trigger"],
        t["trigger_opts"],
        label_visibility="collapsed",
    )
    trigger_details = ""
    none_trigger = t["trigger_opts"][0]
    if trigger_factor == none_trigger:
        trigger_details = st.text_input(
            t["trigger_detail"],
            placeholder=t["trigger_detail_ph"],
        )

    missing = []
    if not description.strip():
        missing.append(t["describe"])
    if not location:
        missing.append(t["location"])
    if other_label in location_choice and not location_other.strip():
        missing.append(t["specify_other"])
    if not medical_history:
        missing.append(t["history"])
    if medical_history and none_label not in medical_history:
        if not medical_history_details.strip():
            missing.append(t["specify_condition"])
    if missing:
        st.info(f"{t['please_complete']} {', '.join(missing)}")

    submitted = st.button(t["analyze"], disabled=bool(missing))
    if submitted:
        payload = {
            "description": description,
            "intensity": intensity,
            "trigger_factor": trigger_factor,
            "trigger_details": trigger_details,
            "duration": duration,
            "location": location,
            "medical_history": medical_history,
            "medical_history_details": medical_history_details,
        }

        payload_key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        payload_hash = hashlib.sha256(payload_key.encode("utf-8")).hexdigest()

        st.success(t["response_captured"])
        st.subheader(t["top_reco"])
        with st.spinner(t["computing"]):
            model_name = (
                "paraphrase-multilingual-MiniLM-L12-v2"
                if lang == "Français"
                else "all-MiniLM-L6-v2"
            )
            results = score_user(payload, referential, model_name=model_name)

        if results:
            # Affichage sous forme de bullet points : top 3 spécialités
            score_explain = (
                "**Note sur le score** : le pourcentage affiché correspond à un **score de similarité sémantique** (similarité cosinus) calculé entre l'embedding de votre description et l'embedding des éléments du référentiel (SBERT). Plus le score est élevé, plus le texte est jugé proche."
                if lang == "Français"
                else "**Score note**: the percentage shown is a **semantic similarity score** (cosine similarity) computed between the embedding of your description and the embedding of the referential items (SBERT). Higher means the texts are considered closer."
            )
            st.info(score_explain)
            top_specialties = results[:3]
            st.markdown("**Top 3 Spécialités recommandées :**")
            for idx, item in enumerate(top_specialties, 1):
                st.markdown(f"- **{item['specialite']}**  (score: {round(item['similarity']*100, 1)}%)\n> {item['symptoms']}")

            # Radar chart des spécialités et leur similarité
            import matplotlib.pyplot as plt
            import numpy as np

            labels = [item['specialite'] for item in top_specialties]
            values = [item['similarity']*100 for item in top_specialties]
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            # Boucler pour fermer le polygone
            values += values[:1]
            angles += angles[:1]
            labels += labels[:1]

            fig, ax = plt.subplots(figsize=(3,3), subplot_kw=dict(polar=True))
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles), labels)
            ax.set_ylim(0, 100)
            ax.set_title("Comparaison des spécialités (similarité)")
            st.pyplot(fig)
        
            import pandas as pd
            import seaborn as sns

            # Récupérer les spécialités du top 3
            top_specialties = results[:3]
            top_pathos = [item['specialite'] for item in top_specialties]

            # Charger le référentiel en DataFrame pour faciliter le traitement
            ref_df = pd.read_csv('data/processed/medical_referential.csv', sep=';')
            if lang == "Français":
                ref_df = pd.read_csv('data/processed/medical_referential_fr.csv', sep=';')

            # Filtrer les lignes du référentiel correspondant au top 3
            # Mapping dynamique des colonnes pour FR/EN (doit être AVANT toute boucle d'usage)
            colmap = {k.lower(): k for k in ref_df.columns}
            col_speciality = colmap.get("speciality") or colmap.get("specialite")
            col_symptoms = colmap.get("symptoms") or colmap.get("symptomes")

            # On ne filtre plus que sur la spécialité
            rows = []
            for spec in top_pathos:
                row = ref_df[ref_df[col_speciality] == spec]
                if not row.empty:
                    rows.append(row.iloc[0])

            # Extraire tous les symptômes uniques du top 3
            all_symptoms = set()
            symptoms_per_patho = []
            for row in rows:
                symptoms = [s.strip() for s in str(row[col_symptoms]).split(',')]
                symptoms_per_patho.append(symptoms)
                all_symptoms.update(symptoms)
            all_symptoms = sorted(all_symptoms)

            import numpy as np
            # Construire la matrice (score de similarité par symptôme)
            spec_to_similarity = {item["specialite"]: float(item["similarity"]) for item in top_specialties}
            data = []
            for row, symptoms in zip(rows, symptoms_per_patho):
                sim = spec_to_similarity.get(row[col_speciality], np.nan)
                data.append([sim if s in symptoms else np.nan for s in all_symptoms])

            heatmap_df = pd.DataFrame(
                data,
                index=[f"{row[col_speciality]}" for row in rows],
                columns=all_symptoms,
            )

            # Afficher la heatmap
            import matplotlib.pyplot as plt
            plt.figure(figsize=(min(12, 1 + len(all_symptoms) * 0.6), 2 + len(rows)))
            if heatmap_df.empty or len(heatmap_df.columns) == 0 or len(heatmap_df.index) == 0:
                st.info("Aucune donnée symptomatique à afficher pour la heatmap des top 3 spécialités.")
            else:
                sns.heatmap(heatmap_df, cmap='Reds', cbar=True, linewidths=.5, fmt='.2f')
                plt.title(
                    "Matrice des symptômes pour le top 3 des spécialités"
                    if lang == "Français"
                    else "Matrix of symptoms for the top 3 specialties"
                )
                plt.xlabel("Symptômes")
                plt.ylabel("Spécialité")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(plt.gcf())
            # Generation (G) - Gemini response
            st.subheader(t["ai_analysis"])
            cached_hash = st.session_state.get("gemini_payload_hash")
            cached_response = st.session_state.get("gemini_response")
            if cached_hash == payload_hash and cached_response:
                st.markdown(cached_response)
            else:
                with st.spinner(t["generating"]):
                    try:
                        lang_code = "fr" if lang == "Français" else "en"
                        augmented_prompt = build_augmented_prompt(payload, results, lang=lang_code)
                        gemini_response = generate_gemini_response(augmented_prompt)
                        st.session_state["gemini_payload_hash"] = payload_hash
                        st.session_state["gemini_response"] = gemini_response
                        st.markdown(gemini_response)
                    except Exception as e:
                        st.error(f"{t['gemini_error']} ({e})")
        else:
            st.warning(t["longer_desc"])

        disclaimer_text = (
        "**Avertissement**: Cet outil est fourni à titre informatif uniquement. Il ne remplace pas un diagnostic médical professionnel. En cas de symptômes sévères ou persistants, consultez un médecin sans délai."
        if lang == "Français"
        else "**Warning**: This tool is for informational purposes only. It is not a substitute for professional medical diagnosis. If you experience severe or persistent symptoms, consult a doctor immediately."
    )
    st.warning(disclaimer_text)
    # Footer
    st.markdown("---")
    st.caption(t["footer"])



if __name__ == "__main__":
    main()
