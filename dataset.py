import os
import json
import torch
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class LongitudinalMIMICDataset(Dataset):
    """
    Custom PyTorch dataset for MIMIC patient time series modeling.
    This class filters the data based on the dataset_splits.json file,
    cleans administrative noise, filters empty targets, and produces padded outputs.
    """
    def __init__(self, data_dir, split_name='train', split_file='dataset_splits.json', tokenizer=None, max_visits=5, transform=None):
        self.data_dir = data_dir
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.max_visits = max_visits
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        #Reading Splits File
        split_path = os.path.join(data_dir, split_file)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found at {split_path}. Please run the split script first.")
            
        with open(split_path, 'r', encoding='utf-8') as f:
            splits = json.load(f)
            
        if split_name not in splits:
            raise ValueError(f"Invalid split_name: {split_name}. Must be 'train', 'val', or 'test'.")
            
        allowed_patients = set(splits[split_name])
        print(f"Loading {split_name} split: Found {len(allowed_patients)} allowed patients.")
        
        #Filter trajectory files & Apply Quality Check (Remove Empty Targets)
        self.patient_files = []
        invalid_count = 0
        
        for root, dirs, files in os.walk(data_dir):
            if "trajectory.json" in files:
                patient_folder = os.path.basename(root)
                if patient_folder in allowed_patients:
                    traj_path = os.path.join(root, "trajectory.json")
                    
                    #QUALITY CHECK
                    with open(traj_path, 'r', encoding='utf-8') as f:
                        patient_data = json.load(f)
                        
                    visits = patient_data.get('visits', [])
                    if len(visits) > 0:
                        last_visit = visits[-1]
                        raw_report = last_visit.get('target_report') or ""
                        
                        if len(raw_report.strip()) > 20:
                            self.patient_files.append(traj_path)
                        else:
                            invalid_count += 1
                    else:
                        invalid_count += 1
                    
        print(f"Successfully loaded {len(self.patient_files)} trajectories for {split_name}.")
        if invalid_count > 0:
            print(f"[DATA CLEANSING] Removed {invalid_count} trajectories due to empty or invalid target reports.")

    def __len__(self):
        return len(self.patient_files)

    def _load_image(self, image_path):
        """Load image or return zero tensor (black image) if there is no image in that view"""
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert('RGB')
                return self.transform(img)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                
        return torch.zeros((3, 224, 224))

    def _parse_report(self, report_text):
        """
        This takes the raw report and returns its parts in a dictionary.
        """
        sections = {}
        if not report_text:
            return sections
            
        pattern = re.compile(r'(?:\n|^)\s*([A-Z][A-Z0-9\s/]{2,}):')
        matches = list(pattern.finditer(report_text))
        
        if not matches:
            sections['UNFORMATTED'] = report_text.strip()
            return sections
            
        for i, match in enumerate(matches):
            header = match.group(1).strip()
            start_idx = match.end()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(report_text)
            content = report_text[start_idx:end_idx].strip()
            sections[header] = content
            
        return sections

    def __getitem__(self, idx):
        with open(self.patient_files[idx], 'r', encoding='utf-8') as f:
            patient_data = json.load(f)

        original_visits = patient_data.get('visits', [])
        total_visits_in_history = len(original_visits)
        
        visits = original_visits[-self.max_visits:] 
        seq_len = len(visits)
        start_offset = total_visits_in_history - seq_len
        
        frontal_imgs = []
        lateral_imgs = []
        delta_ts = []
        clinical_features = []
        text_contexts = []
        prompt_texts = []   
        target_reports = [] 
        
        time_mask = [1] * seq_len + [0] * (self.max_visits - seq_len)
        
        #ADDED: Demographics
        demographics = patient_data.get('demographics', {})
        gender = demographics.get('gender', 'Unknown')
        ethnicity = demographics.get('ethnicity', 'Unknown')

        for i, visit in enumerate(visits):
            true_visit_index = start_offset + i
            
            frontal_imgs.append(self._load_image(visit.get('images', {}).get('frontal')))
            lateral_imgs.append(self._load_image(visit.get('images', {}).get('lateral')))
            
            delta_ts.append(torch.tensor(visit.get('delta_t_days', 0.0), dtype=torch.float32))
            
            vitals = visit.get('vitals')
            if vitals:
                clinical_vec = [
                    float(vitals.get('heartrate', 0) or 0), 
                    float(vitals.get('resprate', 0) or 0), 
                    float(vitals.get('o2sat', 0) or 0), 
                    float(vitals.get('sbp', 0) or 0), 
                    float(vitals.get('temperature', 0) or 0), 
                    float(vitals.get('pain', 0) or 0)
                ]
            else:
                clinical_vec = [0.0] * 6 
                
            clinical_features.append(torch.tensor(clinical_vec, dtype=torch.float32))
            
            text_inputs = visit.get('text_inputs', {})
            meds = text_inputs.get('home_medications') or []
            context_str = (
                f"Demographics: Gender {gender}, Ethnicity {ethnicity}. "
                f"Chief Complaint: {text_inputs.get('chief_complaint') or 'None'}. "
                f"Diagnosis: {text_inputs.get('icd_title') or 'None'}. "
                f"Medications: {', '.join(meds) if meds else 'None'}. "
                f"Previous Report: {text_inputs.get('previous_report') or 'None'}."
            )
            text_contexts.append(context_str)
            
            raw_report = visit.get('target_report') or ""
            sections = self._parse_report(raw_report)
            
            #ROMPT CREATION
            prompt_text = (
                f"Patient Context Data: Gender {gender}, Ethnicity {ethnicity}. "
            )

            if true_visit_index == 0:
                prompt_text += "(This is the patient's first visit.)"
            else:
                prompt_text += (f"Previous Report: {text_inputs.get('previous_report') or 'None'}.")

            #TARGET CREATION
            target_text = ""
            if "FINDINGS" in sections:
                target_text += f"FINDINGS:\n{sections['FINDINGS']}\n\n"
            if "IMPRESSION" in sections:
                target_text += f"IMPRESSION:\n{sections['IMPRESSION']}\n"

            if not target_text and "UNFORMATTED" in sections:
                target_text = sections['UNFORMATTED']
            elif "CONCLUSION" in sections:
                target_text += f"IMPRESSION:\n{sections['CONCLUSION']}\n"
            elif "COMMENTS" in sections:
                target_text += f"IMPRESSION:\n{sections['COMMENTS']}\n"
            
            if not target_text.strip():
                target_text = raw_report.strip()     
                
            #CLEANING
            target_text = re.sub(r'(_+.*?_+)+', '', target_text)
            target_text = re.sub(r'\s+', ' ', target_text)
            target_text = target_text.strip()

            prompt_texts.append(prompt_text.strip())
            target_reports.append(target_text)

        # ---------------------------------------------------------
        pad_len = self.max_visits - seq_len
        if pad_len > 0:
            for _ in range(pad_len):
                frontal_imgs.append(torch.zeros((3, 224, 224)))
                lateral_imgs.append(torch.zeros((3, 224, 224)))
                delta_ts.append(torch.tensor(0.0, dtype=torch.float32))
                clinical_features.append(torch.zeros((6,), dtype=torch.float32))
                text_contexts.append("")
                prompt_texts.append("")   
                target_reports.append("") 

        trajectory_data = {
            "frontal_imgs": torch.stack(frontal_imgs),
            "lateral_imgs": torch.stack(lateral_imgs),
            "delta_t": torch.stack(delta_ts),
            "clinical_features": torch.stack(clinical_features),
            "text_contexts": text_contexts,
            "prompt_texts": prompt_texts,      
            "target_reports": target_reports,  
            "time_mask": torch.tensor(time_mask, dtype=torch.bool)
        }

        return trajectory_data
