import os
import torch
import wandb
import gc


os.environ["WANDB_MODE"] = "offline"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from dataset import LongitudinalMIMICDataset
from models.main_model import MedicalLongitudinalLLM

def train_joint():
    #Configuration for Joint Training
    config = {
        "num_epochs": 20,           
        "batch_size": 1,
        "grad_accum_steps": 4,
        "lr_lora": 1e-4,      
        "lr_modules": 1e-3,   
        "max_visits": 7,     
        "embed_dim": 768,
        "llm_name": "./offline_models/BioMistral-7B",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    
    # ==========================================
    # 2. Data Preparation
    # ==========================================
    print("[INFO] Loading Longitudinal Datasets...")
    train_dataset = LongitudinalMIMICDataset(data_dir="./trajectories", split_name='train', max_visits=config["max_visits"])
    val_dataset = LongitudinalMIMICDataset(data_dir="./trajectories", split_name='val', max_visits=config["max_visits"])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    #3.Model Initialization
    print("[INFO] Initializing Master Model...")
    model = MedicalLongitudinalLLM(llm_name=config["llm_name"], embed_dim=config["embed_dim"], freeze_vision=True)
    model.to(config["device"])


    # 4.1. Joint Training Parameter Setup (Crucial)
    print("[INFO] Configuring Trainable Parameters (Joint Mode)...")
    
    #First,freeze everything
    for name, param in model.named_parameters():
        param.requires_grad = False

        # A) nfreeze LoRA parameters
        if "lora" in name:
            param.requires_grad = True
            
        #B.Unfreeze all bridge/alignment modules (Scratch Modules)
        if any(keyword in name for keyword in [
            "projector", 
            "temporal_memory", 
            "fusion_module", 
            "tabular_encoder.mlp", 
            "text_encoder.proj"
        ]):
            param.requires_grad = True

        #C)Explicitly ensure heavy backbones remain frozen to prevent catastrophic forgetting
        if "vision_encoder.extractor.features" in name or "vision_encoder.extractor.norm" in name:
            param.requires_grad = False
        if "text_encoder.bert" in name:
            param.requires_grad = False

    # Count trainable parameters
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SUCCESS] Trainable Parameters: {trainable_params_count:,}")


    # 4.2. Calculate and Print Trainable Parameters Log
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print("\n" + "="*60)
    print("MODEL PARAMETERS REPORT:")
    print("="*60)
    print(f"#Total Parameters:     {total_params:,}")
    print(f"#Trainable Parameters: {trainable_params:,}")
    print(f"#Trainable Percentage: {100 * trainable_params / total_params:.4f}%")
    print("="*60 + "\n")


    # 5. Differential Optimizer & Scheduler
    # Separate parameters based on their required learning speed
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
    scratch_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" not in n]

    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': config["lr_lora"]},         # LLM gets smaller steps
        {'params': scratch_params, 'lr': config["lr_modules"]}    # New modules learn faster
    ], weight_decay=0.06) #weight_decay=0.01

    all_trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_steps = (len(train_loader) // config["grad_accum_steps"]) * config["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    #scaler = GradScaler()
    scaler = torch.amp.GradScaler('cuda')

    #==========================================
    #6. Main Training Loop
    print("\n[INFO] Starting Joint Training Process...")
    
    for epoch in range(config["num_epochs"]):
        # A. TRAINING PHASE
        model.train()
        total_train_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [TRAIN]")
        
        for step, batch in enumerate(pbar):
            # ================= DEBUG =================
            # if step >= 5: 
            #     break
            # =========================================
            frontal_imgs = batch['frontal_imgs']
            lateral_imgs = batch['lateral_imgs']
            clinical_feats = batch['clinical_features']
            text_contexts = batch['text_contexts']
            delta_t = batch['delta_t']
            time_mask = batch['time_mask']
            
            B_size = frontal_imgs.shape[0]
            target_reports_raw = batch['target_reports']
            prompt_texts_raw = batch['prompt_texts']
            
            last_valid_indices = (batch['time_mask'].sum(dim=1).long() - 1).clamp(min=0)
            
            target_reports = []
            prompt_texts = []

            # Extract the correct target report based on the actual trajectory length
            if isinstance(target_reports_raw, list) and len(target_reports_raw) > 0 and isinstance(target_reports_raw[0], (tuple, list)):
                for b in range(B_size):
                    true_last_idx = last_valid_indices[b].item()
                    target_reports.append(target_reports_raw[true_last_idx][b])
                    prompt_texts.append(prompt_texts_raw[true_last_idx][b])
            elif isinstance(target_reports_raw, list) and len(target_reports_raw) == B_size:
                target_reports = target_reports_raw
                prompt_texts = prompt_texts_raw
            else:
                target_reports = target_reports_raw[:B_size]
                prompt_texts = prompt_texts_raw[:B_size]

            valid_reports = [r for r in target_reports if isinstance(r, str) and len(r.strip()) > 5]        
            if len(valid_reports) != B_size:
                continue

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    frontal_imgs, lateral_imgs, clinical_feats, 
                    text_contexts, delta_t, time_mask, 
                    prompt_texts=prompt_texts, target_texts=target_reports
                )
                loss = outputs.loss / config["grad_accum_steps"]
                
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation & optimization step
            if (step + 1) % config["grad_accum_steps"] == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                # Scope bug strictly fixed here using all_trainable_params
                torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
            total_train_loss += loss.item() * config["grad_accum_steps"]
            pbar.set_postfix({"Loss": f"{loss.item() * config['grad_accum_steps']:.4f}"})
            
            del outputs, loss

        avg_train_loss = total_train_loss / len(train_loader)

        #B)VALIDATION PHASE
        model.eval()
        total_val_loss = 0
        logged_samples = 0
        val_table = wandb.Table(columns=["Global Epoch", "Input Prompt", "Target (Ground Truth)", "Generated Report"])
        
        print(f"\n[INFO] Running Validation for Epoch {epoch+1}...")
        
        with torch.no_grad():
            for val_step, batch in enumerate(tqdm(val_loader, desc="[VALIDATION]")):
                # ================= DEBUG =================
                # if val_step >= 2:
                #     break
                # =========================================
                frontal_imgs = batch['frontal_imgs']
                lateral_imgs = batch['lateral_imgs']
                clinical_feats = batch['clinical_features']
                text_contexts = batch['text_contexts']
                delta_t = batch['delta_t']
                time_mask = batch['time_mask']
                
                B_size = frontal_imgs.shape[0]
                target_reports_raw = batch['target_reports']
                prompt_texts_raw = batch['prompt_texts']
                
                last_valid_indices = (batch['time_mask'].sum(dim=1).long() - 1).clamp(min=0)
                
                target_reports = []
                prompt_texts = []

                if isinstance(target_reports_raw, list) and len(target_reports_raw) > 0 and isinstance(target_reports_raw[0], (tuple, list)):
                    for b in range(B_size):
                        true_last_idx = last_valid_indices[b].item()
                        target_reports.append(target_reports_raw[true_last_idx][b])
                        prompt_texts.append(prompt_texts_raw[true_last_idx][b])
                elif isinstance(target_reports_raw, list) and len(target_reports_raw) == B_size:
                    target_reports = target_reports_raw
                    prompt_texts = prompt_texts_raw
                else:
                    target_reports = target_reports_raw[:B_size]
                    prompt_texts = prompt_texts_raw[:B_size]

                valid_reports = [r for r in target_reports if isinstance(r, str) and len(r.strip()) > 5]        
                if len(valid_reports) != B_size:
                    continue

                # Validation Loss Calculation
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(
                        frontal_imgs, lateral_imgs, clinical_feats, 
                        text_contexts, delta_t, time_mask, 
                        prompt_texts=prompt_texts, target_texts=target_reports
                    )
                    val_loss = outputs.loss
                
                total_val_loss += val_loss.item()
                del outputs, val_loss

                # W&B Generation Logging (Top 10 samples)
                if logged_samples < 10:
                    needed = 10 - logged_samples
                    take_n = min(needed, B_size)

                    if isinstance(text_contexts, (list, tuple)) and len(text_contexts) > 0 and isinstance(text_contexts[0], (list, tuple)):
                        sliced_text_contexts = [t_item[:take_n] for t_item in text_contexts]
                    else:
                        sliced_text_contexts = text_contexts[:take_n]
                    
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        generated_reports = model.generate_report(
                            frontal_imgs=frontal_imgs[:take_n], 
                            lateral_imgs=lateral_imgs[:take_n], 
                            clinical_feats=clinical_feats[:take_n], 
                            text_contexts=sliced_text_contexts,
                            delta_t=delta_t[:take_n], 
                            time_mask=time_mask[:take_n], 
                            instruction_text=prompt_texts[:take_n],
                            max_new_tokens=150
                        )
                    
                    for i in range(take_n):
                        val_table.add_data(
                            epoch + 1,
                            prompt_texts[i],
                            target_reports[i],
                            generated_reports[i]
                        )
                    logged_samples += take_n

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"--> Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ------------------------------------------
        # C. W&B LOGGING & CHECKPOINTING
        # ------------------------------------------
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "val/generation_samples": val_table
        })
        
        # Save complete checkpoint
        save_dir = f"./checkpoints_joint/epoch_{epoch+1}"
        os.makedirs(save_dir, exist_ok=True)
        
        model.llm.save_pretrained(save_dir)
        torch.save(model.projector.state_dict(), f"{save_dir}/projector.pth")
        torch.save(model.temporal_memory.state_dict(), f"{save_dir}/temporal_memory.pth")
        torch.save(model.vision_encoder.state_dict(), os.path.join(save_dir, "vision_encoder.pth"))
        torch.save(model.tabular_encoder.state_dict(), os.path.join(save_dir, "tabular_encoder.pth"))
        torch.save(model.text_encoder.state_dict(), os.path.join(save_dir, "text_encoder.pth"))

        # Optional: Save a quick TXT log
        with open("joint_training_log.csv", "a") as f:
            if os.stat("joint_training_log.csv").st_size == 0 if os.path.exists("joint_training_log.csv") else True:
                f.write("Epoch,Train_Loss,Val_Loss\n")
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

        print(f"[CLEANUP] Freeing VRAM after Epoch {epoch+1}...")
        gc.collect()
        torch.cuda.empty_cache()

    wandb.finish()
    print("[SUCCESS] Joint Training Complete!")

if __name__ == "__main__":
    train_joint()