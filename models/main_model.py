import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

# Import all the modules
from models.vision_encoder import MultiViewVisionEncoder
from models.tabular_encoder import ClinicalTabularEncoder
from models.text_encoder import ClinicalTextEncoder
from models.temporal_memory import LongitudinalPatientMemory
from models.multimodal_projector import MultimodalProjector

class MedicalLongitudinalLLM(nn.Module):
    """
        End-to-End Model.
        This class integrates all encoders and injects the output vectors into the LLM as "Soft Prompts".
    """
    def __init__(self, llm_name="BioMistral-7B", embed_dim=768, freeze_vision=True):
        super().__init__()
        
        #1. Initialization Phase One: Encoders
        self.vision_encoder = MultiViewVisionEncoder(embed_dim=embed_dim, freeze_base=freeze_vision)
        self.tabular_encoder = ClinicalTabularEncoder(input_dim=6, embed_dim=embed_dim)
        self.text_encoder = ClinicalTextEncoder(embed_dim=embed_dim, freeze_base=True)
        
        #2. Phase 2 initialization: Longitudinal memory
        self.temporal_memory = LongitudinalPatientMemory(embed_dim=embed_dim)
        
        #3. Loading LLM

        print(f"Loading Base LLM ({llm_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, 
            torch_dtype=torch.bfloat16, #torch_dtype=torch.bfloat16, quantization_config=bnb_config
            device_map="auto",
            use_safetensors=False #added
        )
        
        #Extract the dimensions of the LLM input
        self.llm_dim = self.llm.config.hidden_size
        
        #4. Phase 3 initialization: Projector
        self.projector = MultimodalProjector(input_dim=embed_dim, llm_dim=self.llm_dim)
        
        #5. Applying LoRA to LLM (unlocking 1% of weights for training)
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "v_proj", "k_proj","gate_proj","up_proj", "down_proj"], # Apply to Attention layers
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters() # Print the percentage of trainable parameters

    def forward(self, frontal_imgs, lateral_imgs, clinical_feats, text_contexts, delta_t, time_mask, prompt_texts, target_texts):
        """
        Forward execution of the entire system with ignore_index Masking.
        prompt_texts: 
        target_texts:
        """
        device = self.llm.device
        #cleaning
        clinical_feats = torch.nan_to_num(clinical_feats, nan=0.0)
        frontal_imgs = torch.nan_to_num(frontal_imgs, nan=0.0)
        lateral_imgs = torch.nan_to_num(lateral_imgs, nan=0.0)
        
        # --- Phase 1: Feature Extraction ---
        vision_feat = self.vision_encoder(frontal_imgs.to(device), lateral_imgs.to(device))
        tabular_feat = self.tabular_encoder(clinical_feats.to(device))

        #fixed
        B, T = vision_feat.shape[0], vision_feat.shape[1]
        
        #clean text contexts
        flat_clean_texts = []
        if isinstance(text_contexts, (list, tuple)) and len(text_contexts) > 0 and isinstance(text_contexts[0], (tuple, list)):
            for b in range(B):
                for t in range(T):
                    raw_text = text_contexts[t][b]
                    if raw_text is None or str(raw_text).lower() == 'nan':
                        flat_clean_texts.append("")
                    else:
                        flat_clean_texts.append(str(raw_text))
        else:
            for raw_text in text_contexts:
                if raw_text is None or str(raw_text).lower() == 'nan':
                    flat_clean_texts.append("")
                else:
                    flat_clean_texts.append(str(raw_text))
                    
        text_contexts = flat_clean_texts   

        text_feat = self.text_encoder(text_contexts) # It will be transferred to the device itself
        
        #Restore the time dimension to the text output (B, T, D)
        text_feat = text_feat.view(B, T, -1)
        
        #--- Phase 2: Longitudinal Memory (Time Injection) ---
        patient_state = self.temporal_memory(
            vision_feat, tabular_feat, text_feat, 
            delta_t.to(device), time_mask.to(device)
        )
        
        #Extract the final patient state (last actual visit based on mask)
        final_state = patient_state[:, -1, :] #dimension:(B, 768)
        
        #--- Phase 3: Projector (LLM translation) ---
        soft_prompt_embeds = self.projector(final_state) #.unsqueeze(1) 
        

        #Phase 4: Combine with LLM & APPLY ignore_index
    
        # 1.Pasting a prompt and report together to enter the model
        full_texts = [f"{p}\n\n{t}{self.tokenizer.eos_token}" for p, t in zip(prompt_texts, target_texts)]
        
        #2.full text tokenization
        tokenized_full = self.tokenizer(
            full_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # 3.tokenizing the prompt alone (just to find its length)
        tokenized_prompt = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        input_ids = tokenized_full.input_ids
        text_attention_mask = tokenized_full.attention_mask
        
        #4.Copying IDs to create error calculation labels
        text_labels = input_ids.clone()
        
        #Applying ignore_index to individual batches
        for i in range(B):
            #A: Find the length of the prompt in this sentence.
            prompt_len = (tokenized_prompt.attention_mask[i] == 1).sum().item()
            
            #B: Set the prompt section error to zero (convert to -100)
            text_labels[i, :prompt_len] = -100
            
            # C: Zeroing the padding token error (white space at the end of short sentences)
            padding_mask = input_ids[i] == self.tokenizer.pad_token_id
            text_labels[i, padding_mask] = -100

        # 5.Converting words into embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        #6.Pasting patient history (Soft Prompt) to the beginning of the text
        combined_embeds = torch.cat([soft_prompt_embeds, text_embeds], dim=1)
        
        #7.Masking the Soft Prompt itself in the labels (the model is not supposed to predict the image vector)
        num_soft_tokens = soft_prompt_embeds.shape[1]
        soft_labels = torch.full((B, num_soft_tokens), -100, dtype=torch.long, device=device)
        labels = torch.cat([soft_labels, text_labels], dim=1)
        
        # 8.Making the final Attention Mask
        soft_mask = torch.ones((B, num_soft_tokens), dtype=torch.long, device=device)
        combined_mask = torch.cat([soft_mask, text_attention_mask], dim=1)
        
        # 9.Optimized Loss Calculation
        outputs = self.llm(
            inputs_embeds=combined_embeds, 
            attention_mask=combined_mask,
            labels=labels
        )
        
        return outputs

    def generate_report(self, frontal_imgs, lateral_imgs, clinical_feats, text_contexts, delta_t, time_mask, instruction_text, max_new_tokens=100, temperature=0.7, top_p=0.9):
        """
        This function is used for test time (Inference).
        Instead of calculating Loss, this function produces the final report verbatim.
        """
        self.eval()
        device = self.llm.device
        
        with torch.no_grad(): 
            # --- Phase 1: Feature Extraction ---
            vision_feat = self.vision_encoder(frontal_imgs.to(device), lateral_imgs.to(device))
            tabular_feat = self.tabular_encoder(clinical_feats.to(device))

            #Extraction of B and T moved up
            B, T = vision_feat.shape[0], vision_feat.shape[1]

            #clean text contexts
            flat_clean_texts = []

            if isinstance(text_contexts, (list, tuple)) and len(text_contexts) > 0 and isinstance(text_contexts[0], (tuple, list)):
                for b in range(B):
                    for t in range(T):
                        raw_text = text_contexts[t][b]
                        if raw_text is None or str(raw_text).lower() == 'nan':
                            flat_clean_texts.append("")
                        else:
                            flat_clean_texts.append(str(raw_text))
            else:
                for raw_text in text_contexts:
                    if raw_text is None or str(raw_text).lower() == 'nan':
                        flat_clean_texts.append("")
                    else:
                        flat_clean_texts.append(str(raw_text))
                    
            text_contexts = flat_clean_texts

            text_feat = self.text_encoder(text_contexts)
            
            text_feat = text_feat.view(B, T, -1)
            
            #- Phase 2: Longitudinal Memory ---
            patient_state = self.temporal_memory(
                vision_feat, tabular_feat, text_feat, 
                delta_t.to(device), time_mask.to(device)
            )
            final_state = patient_state[:, -1, :] # Extract the final vector
            
            #--- Phase 3: Projector ---
            soft_prompt_embeds = self.projector(final_state) #.unsqueeze(1) 
            
            #--- Phase 4: Preparing the text prompt ---
            text_inputs = self.tokenizer(instruction_text, return_tensors="pt", padding=True).to(device)
            text_embeds = self.llm.get_input_embeddings()(text_inputs['input_ids'])
            
            #Combine patient history token with text command tokens
            combined_embeds = torch.cat([soft_prompt_embeds, text_embeds], dim=1)
            
            
            #Create Attention Mask (Dynamic length based on N tokens)
            num_soft_tokens = soft_prompt_embeds.shape[1]
            multimodal_mask = torch.ones((B, num_soft_tokens), dtype=torch.long, device=device)
            combined_mask = torch.cat([multimodal_mask, text_inputs['attention_mask']], dim=1)
    
            #--- Phase 5: Text Generation --
            outputs = self.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=50,           
                do_sample=False,              
                num_beams=1,                   
                no_repeat_ngram_size=3,        
                length_penalty=1.2,            
                early_stopping=True,

                use_cache=True,                
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
         
            generated_reports = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            return generated_reports
