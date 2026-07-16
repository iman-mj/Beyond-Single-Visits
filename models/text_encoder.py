import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ClinicalTextEncoder(nn.Module):
    """
    A text encoder for clinical data.
    This module takes combined texts (previous report, diagnosis, medications),
    converts them into tokens using ClinicalBERT, and 
    extracts their dense and meaningful embedding vectors.
    """
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", embed_dim=768, freeze_base=True):
        super().__init__()
        
        #Loading the tokenizer and base model from HuggingFace
        #this model is trained exactly on MIMIC texts.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        #Freezing the BERT main body(PEFT strategy)
        #just want BERT to act as a "feature extractor"
        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        #Projector Layer (learnable)
        #The BERT output is 768. Even if we want to have another embed_dim,
        # This layer adjusts it and, more importantly, learns to align text with images.
        self.proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, text_batch):
        """
        Input:
            text_batch: In DataLoader, texts are output as a list of sequences.
                       So I input it as a one-dimensional list (Flat List) of strings
                        of length (B * T).
        """
        #Detect device
        device = next(self.proj.parameters()).device
        
        #1.Dynamic tokenization of texts
        #padding=True causes short texts to be padded with zeros.
        #truncation=True cuts very long texts (more than 512 tokens) to avoid filling up VRAM.
        encoded_inputs = self.tokenizer(
            text_batch, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        
        #2.passing from ClinicalBERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        #3.Vector extraction [CLS]
        #In the BERT architecture, the first output token (index 0), called CLS, represents the context of the entire sentence.
        #The dimensions of cls_representation will be (B * T, 768)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        
        #4.Passing through the projector
        projected_text = self.proj(cls_representation)
        
        return projected_text


#Sanity Check
if __name__ == "__main__":
    # Simulating the text output of the DataLoader (assuming Batch Size=2 and Max Visits=2)
    #--
    dummy_texts = [
        "Chief Complaint: Dyspnea. Diagnosis: Pneumonia. Medications: None.",
        "Chief Complaint: Follow up. Diagnosis: Pneumonia. Previous Report: Edema improved.",
        "Chief Complaint: Chest pain. Diagnosis: Heart failure. Medications: Diuretics.",
        "" # Blank (padded) visit
    ]
    
    #create model
    print("Loading ClinicalBERT... (This might take a minute the first time)")
    text_encoder = ClinicalTextEncoder(embed_dim=768)
    
    #To test, we put the model into evaluation mode.
    text_encoder.eval()
    
    #Data transfer
    with torch.no_grad():
        output_features = text_encoder(dummy_texts)
        
        #Restore the time dimension (T)
        # (This is done in the main Phase 2 class, but is included here to demonstrate the logic)
        B, T = 2, 2
        final_output = output_features.view(B, T, -1)
    
    print(f"Input List Length: {len(dummy_texts)} (B * T)")
    print(f"Output Features Shape: {final_output.shape}") 
    #  (2, 2, 768)