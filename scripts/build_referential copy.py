import csv
from collections import defaultdict
from pathlib import Path


FIX_DISEASE = {
    "Dimorphic hemmorhoids(piles)": "Dimorphic hemorrhoids(piles)",
    "hepatitis A": "Hepatitis A",
}

FIX_SPECIALTY = {
    "Internal Medcine": "Internal Medicine",
    "Rheumatologists": "Rheumatologist",
    "Dermatologists": "Dermatologist",
    "hepatologist": "Hepatologist",
}


def read_csv(path, encoding):
    with path.open("r", encoding=encoding, newline="") as handle:
        return list(csv.reader(handle))


def norm(text, fixes=None, lower=False):
    text = " ".join(text.strip().split())
    if fixes:
        text = fixes.get(text, text)
    if lower:
        text = text.lower()
    return text


def build_referential(raw_dir):
    original_rows = read_csv(raw_dir / "Original_Dataset_FR.csv", "utf-8-sig")
    description_rows = read_csv(raw_dir / "Disease_Description_FR.csv", "utf-8-sig")
    mapping_rows = read_csv(raw_dir / "Doctor_Versus_Disease_FR.csv", "utf-8")

    symptoms_by_disease = defaultdict(set)
    for row in original_rows[1:]:
        if not row:
            continue
        disease = norm(row[0], FIX_DISEASE)
        for symptom in row[1:]:
            symptom = norm(symptom.replace("_", " "), lower=True)
            if symptom:
                symptoms_by_disease[disease].add(symptom)

    descriptions = {}
    for row in description_rows[1:]:
        if not row:
            continue
        disease = norm(row[0], FIX_DISEASE)
        desc = row[1].strip() if len(row) > 1 else ""
        if desc:
            descriptions[disease] = desc

    specialty_by_disease = {}
    for row in mapping_rows:
        if len(row) < 2:
            continue
        disease = norm(row[0], FIX_DISEASE)
        specialty_by_disease[disease] = norm(row[1], FIX_SPECIALTY)

    rows = []
    for disease, specialty in sorted(specialty_by_disease.items()):
        symptoms = set(symptoms_by_disease.get(disease, set()))
        rows.append(
            {
                "Speciality": specialty,
                "Disease": disease,
                "Symptoms": ", ".join(sorted(symptoms)),
                "Description": descriptions.get(disease, ""),
            }
        )
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = build_referential(Path("data/raw"))
    write_csv(rows, Path("data/processed/medical_referential_fr.csv"))
    print(f"Wrote medical_referential_fr.csv with {len(rows)} rows.")


if __name__ == "__main__":
    main()
