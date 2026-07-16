import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

BASE_MIMIC_DIR = "./mimic-iv"
OUTPUT_DIR = "./processed_trajectories"

PATHS = {
    'reports_dir': os.path.join(BASE_MIMIC_DIR, 'mimic-cxr-reports', 'files'),
    'medrecon': os.path.join(BASE_MIMIC_DIR, 'mimic-iv-ed-2.2', 'ed', 'medrecon.csv'),
    'diagnoses': os.path.join(BASE_MIMIC_DIR, 'mimic-iv-3.1', 'hosp', 'diagnoses_icd.csv'),
    'd_icd': os.path.join(BASE_MIMIC_DIR, 'mimic-iv-3.1', 'hosp', 'd_icd_diagnoses.csv'),
    'patients': os.path.join(BASE_MIMIC_DIR, 'mimic-iv-3.1', 'hosp', 'patients.csv'),
    'admissions': os.path.join(BASE_MIMIC_DIR, 'mimic-iv-3.1', 'hosp', 'admissions.csv'),
    'triage': os.path.join(BASE_MIMIC_DIR, 'mimic-iv-ed-2.2', 'ed', 'triage.csv'),
    'edstays': os.path.join(BASE_MIMIC_DIR, 'mimic-iv-ed-2.2', 'ed', 'edstays.csv'),
    'cxr_meta': os.path.join(BASE_MIMIC_DIR, 'MIMIC-CXR', 'for_cxr', 'mimic-cxr-2.0.0-metadata.csv'),
    'cxr_images_dir': os.path.join(BASE_MIMIC_DIR, 'MIMIC-CXR', 'official_data_iccv_final', 'files')
}


missing_log = {
    "missing_images": [],
    "missing_clinical_records": [],
    "missing_reports": []
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataframes (Optimized for Memory)...")

df_patients = pd.read_csv(PATHS['patients'], usecols=['subject_id', 'gender'])
df_admissions = pd.read_csv(PATHS['admissions'], usecols=['subject_id', 'hadm_id', 'race'])
df_admissions = df_admissions.rename(columns={'race': 'ethnicity'})
df_triage = pd.read_csv(PATHS['triage'], usecols=['subject_id', 'stay_id', 'chiefcomplaint', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity'])
df_triage['pain'] = pd.to_numeric(df_triage['pain'], errors='coerce')
df_cxr = pd.read_csv(PATHS['cxr_meta'], usecols=['dicom_id', 'subject_id', 'study_id', 'ViewPosition', 'StudyDate', 'StudyTime'])
# df_edstays = pd.read_csv(PATHS['edstays'], usecols=['subject_id', 'stay_id', 'intime', 'outtime'])
df_edstays = pd.read_csv(PATHS['edstays'], usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'])
df_edstays['intime'] = pd.to_datetime(df_edstays['intime'])
df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'])
df_triage = pd.merge(df_triage, df_edstays, on=['subject_id', 'stay_id'], how='inner')

df_diagnoses = pd.read_csv(PATHS['diagnoses'], usecols=['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version'])
df_dicd = pd.read_csv(PATHS['d_icd'], usecols=['icd_code', 'icd_version', 'long_title'])

df_diagnoses = pd.merge(df_diagnoses, df_dicd, on=['icd_code', 'icd_version'], how='left')

df_primary_diag = df_diagnoses[df_diagnoses['seq_num'] == 1]

df_medrecon = pd.read_csv(PATHS['medrecon'], usecols=['subject_id', 'stay_id', 'etcdescription'])

print("Merging data...")

df_demo = pd.merge(df_patients, df_admissions[['subject_id', 'ethnicity']].drop_duplicates(), on='subject_id', how='left')

df_cxr['StudyTime_clean'] = df_cxr['StudyTime'].astype(str).str.split('.').str[0].str.zfill(6)
df_cxr['StudyDateTime'] = pd.to_datetime(
    df_cxr['StudyDate'].astype(str) + ' ' + df_cxr['StudyTime_clean'], 
    format='%Y%m%d %H%M%S', 
    errors='coerce'
)
df_cxr = df_cxr.sort_values(by=['subject_id', 'StudyDateTime'])

grouped_cxr = df_cxr.groupby(['subject_id', 'study_id'])

print("Processing longitudinal trajectories...")

unique_patients = df_cxr['subject_id'].unique()

for subject_id in unique_patients:
    patient_dir = os.path.join(OUTPUT_DIR, f"p{str(subject_id)[:2]}", f"p{subject_id}")
    os.makedirs(patient_dir, exist_ok=True)
    
    patient_demo = df_demo[df_demo['subject_id'] == subject_id].iloc[0] if not df_demo[df_demo['subject_id'] == subject_id].empty else None
    
    patient_trajectory = {
        "subject_id": int(subject_id),
        "demographics": {
            "gender": patient_demo['gender'] if patient_demo is not None else None,
            "ethnicity": patient_demo['ethnicity'] if patient_demo is not None else None
        },
        "visits": []
    }
    
    patient_studies = df_cxr[df_cxr['subject_id'] == subject_id]['study_id'].unique()
    
    time_step = 0
    previous_datetime = None
    previous_report_text = None
    
    for study_id in patient_studies:
        study_records = df_cxr[(df_cxr['subject_id'] == subject_id) & (df_cxr['study_id'] == study_id)]
        current_datetime = study_records['StudyDateTime'].iloc[0]
        
        delta_t_days = 0
        if previous_datetime and pd.notnull(current_datetime) and pd.notnull(previous_datetime):
            delta_t_days = (current_datetime - previous_datetime).total_seconds() / (24 * 3600)
        previous_datetime = current_datetime
        
        images_dict = {"frontal": None, "lateral": None}
        for _, row in study_records.iterrows():
            img_path = os.path.join(PATHS['cxr_images_dir'], f"p{str(subject_id)[:2]}", f"p{subject_id}", f"s{study_id}", f"{row['dicom_id']}.jpg")
            
            if not os.path.exists(img_path):
                missing_log["missing_images"].append(img_path)
            else:
                if row['ViewPosition'] in ['AP', 'PA']:
                    images_dict["frontal"] = img_path
                elif row['ViewPosition'] in ['LATERAL', 'LL']:
                    images_dict["lateral"] = img_path

        patient_triage = df_triage[df_triage['subject_id'] == subject_id]
        
        matched_triage = None
        
        if not patient_triage.empty and pd.notnull(current_datetime):
            exact_match = patient_triage[
                (patient_triage['intime'] <= current_datetime) & 
                (patient_triage['outtime'] >= current_datetime)
            ]
            
            if not exact_match.empty:
                matched_triage = exact_match.iloc[0]
            else:
                time_diffs = (patient_triage['intime'] - current_datetime).abs()
                if time_diffs.min() <= pd.Timedelta(hours=24):
                    matched_triage = patient_triage.loc[time_diffs.idxmin()]

        icd_title = None
        home_medications = []

        if matched_triage is not None and not matched_triage.empty:
            vitals = {
                "temperature": matched_triage['temperature'],
                "heartrate": matched_triage['heartrate'],
                "resprate": matched_triage['resprate'],
                "o2sat": matched_triage['o2sat'],
                "sbp": matched_triage['sbp'],
                "dbp": matched_triage['dbp'],
                "pain": matched_triage['pain'],
                "acuity": matched_triage['acuity']
            }
            chief_complaint = matched_triage['chiefcomplaint']
            current_hadm_id = matched_triage.get('hadm_id')
            if pd.notnull(current_hadm_id):
                diag_row = df_primary_diag[df_primary_diag['hadm_id'] == current_hadm_id]
                if not diag_row.empty:
                    icd_title = diag_row['long_title'].iloc[0]

            current_stay_id = matched_triage.get('stay_id')
            if pd.notnull(current_stay_id):
                meds_series = df_medrecon[df_medrecon['stay_id'] == current_stay_id]['etcdescription']
                home_medications = meds_series.dropna().unique().tolist()

        else:
            missing_log["missing_clinical_records"].append(f"Subject: {subject_id}, Study: {study_id} (No temporal match)")
            vitals = None
            chief_complaint = None

        report_path = os.path.join(PATHS['reports_dir'], f"p{str(subject_id)[:2]}", f"p{subject_id}", f"s{study_id}.txt")
        current_report_text = None
        
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                current_report_text = f.read().strip()
        else:
            missing_log["missing_reports"].append(report_path)


        visit_data = {
            "time_step": time_step,
            "study_id": int(study_id),
            "study_datetime": str(current_datetime),
            "delta_t_days": round(delta_t_days, 2),
            "images": images_dict,
            "vitals": vitals,
            "text_inputs": {
                "chief_complaint": chief_complaint,
                "icd_title": icd_title,
                "home_medications": home_medications,
                "previous_report": previous_report_text
            },
            "target_report": current_report_text
        }
        
        patient_trajectory["visits"].append(visit_data)
        previous_report_text = current_report_text
        time_step += 1
        
    json_path = os.path.join(patient_dir, "trajectory.json")
    patient_trajectory_clean = json.loads(pd.Series(patient_trajectory).to_json(orient='records')) 
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(patient_trajectory, f, indent=4, default=str)

missing_log_path = os.path.join(OUTPUT_DIR, "missing_data_report.json")
with open(missing_log_path, 'w', encoding='utf-8') as f:
    json.dump(missing_log, f, indent=4)

print(f"\nProcessing Complete!")
print(f"Trajectories saved in: {OUTPUT_DIR}")
print(f"Missing data report saved in: {missing_log_path}")
print(f"Total missing images: {len(missing_log['missing_images'])}")
print(f"Total missing clinical records: {len(missing_log['missing_clinical_records'])}")
