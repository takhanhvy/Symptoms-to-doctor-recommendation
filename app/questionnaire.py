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

    with st.form("questionnaire_form"):
        description = st.text_area(
            "Describe your symptoms in your own words",
            placeholder="Example: headache with nausea since yesterday...",
        )
        context = st.text_area(
            "Add any useful context (needs, constraints, history)",
            placeholder="Example: recent travel, medication, allergies, chronic issues...",
        )
        intensity = st.radio(
            "Intensity of pain/discomfort (1 = low, 5 = very high)",
            [1, 2, 3, 4, 5],
            horizontal=True,
        )
        duration = st.selectbox(
            "Duration",
            [
                "Less than 24 hours",
                "1-3 days",
                "4-7 days",
                "1-4 weeks",
                "More than 1 month",
            ],
        )
        location = st.text_input(
            "Location / body area",
            placeholder="Example: head, chest, stomach, lower back...",
        )
                         
        submitted = st.form_submit_button("Analyze")

    if submitted:
        payload = {
            "description": description,
            "context": context,
            "intensity": intensity,
            "duration": duration,
            "location": location,        
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
