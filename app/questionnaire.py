import json
import sys
from pathlib import Path
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.scoring import load_referential, score_user

def main():
    st.set_page_config(page_title="Medical Orientation Questionnaire")
    st.title("Medical Orientation Questionnaire")
    st.warning(
        "**Warning**: This tool is for informational purposes only. It is not a substitute for professional medical diagnosis. If you experience severe or persistent symptoms, consult a doctor immediately."
    )

    referential_path = Path("data/processed/medical_referential.csv")
    referential = load_referential(referential_path)

    def required_label(text):
        st.markdown(f"{text} <span style='color:red'>*</span>", unsafe_allow_html=True)

    required_label("Describe your situation in your own words")
    description = st.text_area(
        "Describe your situation in your own words",
        placeholder="Example: headache with nausea since yesterday...",
        label_visibility="collapsed",
    )

    required_label("Intensity of pain/discomfort (1 = low, 5 = very high)")
    intensity = st.radio(
        "Intensity of pain/discomfort (1 = low, 5 = very high)",
        [1, 2, 3, 4, 5],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    required_label("Duration")
    duration = st.selectbox(
        "Duration",
        [
            "Less than 24 hours",
            "1-3 days",
            "4-7 days",
            "1-4 weeks",
            "More than 1 month",
        ],
        label_visibility="collapsed",
    )

    location_options = [
        "Head / Skull",
        "Eyes",
        "Ears",
        "Nose / Throat",
        "Chest / Thorax",
        "Abdomen / Belly",
        "Back",
        "Arms / Hands",
        "Legs / Feet",
        "Skin",
        "Generalized / Whole Body",
        "Other",
    ]
    required_label("Location / body area (select all that apply)")
    location_choice = st.multiselect(
        "Location / body area (select all that apply)",
        location_options,
        label_visibility="collapsed",
    )
    location_other = ""
    if "Other" in location_choice:
        required_label("Specify other location")
        location_other = st.text_input(
            "Specify other location",
            placeholder="Example: neck, jaw, pelvis...",
            label_visibility="collapsed",
        )
    location = [loc for loc in location_choice if loc != "Other"]
    if location_other.strip():
        location.append(location_other.strip())

    history_options = [
        "Heart problems / Hypertension",
        "Diabetes",
        "Respiratory problems / Asthma",
        "Known allergies",
        "Chronic digestive problems",
        "Autoimmune diseases",
        "Cancer history",
        "Neurological disorders",
        "Kidney problems",
        "Recent surgical operations",
        "Current medication treatment",
        "No particular medical history",
    ]
    def enforce_medical_history():
        selected = st.session_state.get("medical_history", [])
        if "No particular medical history" in selected and len(selected) > 1:
            st.session_state["medical_history"] = ["No particular medical history"]
            st.session_state["medical_history_locked"] = True
        else:
            st.session_state["medical_history_locked"] = False

    required_label(
        "Do you have any of the following medical conditions or risk factors? (multiple choices possible)"
    )
    medical_history = st.multiselect(
        "Do you have any of the following medical conditions or risk factors? (multiple choices possible)",
        history_options,
        key="medical_history",
        on_change=enforce_medical_history,
        label_visibility="collapsed",
    )
    if st.session_state.get("medical_history_locked"):
        st.info("'No particular medical history' was chosen.")
    medical_history_details = ""
    if medical_history and "No particular medical history" not in medical_history:
        required_label("Specify your condition")
        medical_history_details = st.text_input(
            "Specify your condition",
            placeholder="Example: type 2 diabetes, seasonal asthma...",
            label_visibility="collapsed",
        )

    required_label("When do symptoms appear or worsen?")
    trigger_factor = st.selectbox(
        "When do symptoms appear or worsen?",
        [   
            "None of the options",
            "During physical effort / exercise",
            "At rest / during night",
            "After meals",
            "When lying down / standing up",
            "During stressful situations",
            "Upon contact with certain substances (food, allergens...)",
            "In cold / hot weather",
        ],
        label_visibility="collapsed",
    )
    trigger_details = ""
    if trigger_factor == "None of the options":
        trigger_details = st.text_input(
            "Specify if trigger exists",
            placeholder="Example: after long screen time, during travel...",
        )

    missing = []
    if not description.strip():
        missing.append("Description")
    if not location:
        missing.append("Location")
    if "Other" in location_choice and not location_other.strip():
        missing.append("Other location detail")
    if not medical_history:
        missing.append("Medical history")
    if medical_history and "No particular medical history" not in medical_history:
        if not medical_history_details.strip():
            missing.append("Medical history details")
    if missing:
        st.info(f"Please complete: {', '.join(missing)}")

    submitted = st.button("Analyze", disabled=bool(missing))
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

        st.success("Response captured.")
        st.subheader("Top recommendations")
        with st.spinner("Computing similarities..."):
            results = score_user(payload, referential)

        if results:
            st.table(
                [
                    {
                        "Specialty": item["specialite"],
                        "Best-matching disease": item["pathologie"],
                        "Similarity": round(item["similarity"], 3),
                        "Score": round(item["score"], 3),
                    }
                    for item in results
                ]
            )
            for item in results:
                st.caption(f"{item['specialite']}: {item['explanation']}")
        else:
            st.warning("Please provide a longer description to compute recommendations.")


    # Footer
    st.markdown("---")
    st.caption("""
    **EFREI Project - Generative AI & Semantic Analysis 2025**  
    RAG System with SBERT + Gemini 2.0 Flash | Medical Orientation Prototype
    """)


if __name__ == "__main__":
    main()
