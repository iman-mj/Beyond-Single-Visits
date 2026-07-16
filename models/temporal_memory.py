import torch
import torch.nn as nn
import math

class ContinuousTimeEncoder(nn.Module):
    """
    Continuous time encoder.
    Takes a number and converts it into a dense vector
    of the same dimensions as the model (768) to be concatenated with the features.
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        # Building a linear layer to create different frequencies from a scalar number
        self.omega = nn.Linear(1, embed_dim)

    def forward(self, delta_t):
        #input:(B, T)
        #Transform to (B, T, 1) to enter the linear layer
        delta_t = delta_t.unsqueeze(-1)
        
        # Generation of frequencies
        time_hidden = self.omega(delta_t)
        
        #Applying sine and cosine functions alternately
        time_encoding = torch.zeros_like(time_hidden)
        time_encoding[:, :, 0::2] = torch.sin(time_hidden[:, :, 0::2])
        time_encoding[:, :, 1::2] = torch.cos(time_hidden[:, :, 1::2])
        
        return time_encoding


class VisitModalityFusion(nn.Module):
    def __init__(self, embed_dim=768, dropout=0.1):
        super().__init__()

        #---
        total_dim = embed_dim * 3 
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, vision_feat, tabular_feat, text_feat):
        concatenated = torch.cat([vision_feat, tabular_feat, text_feat], dim=-1)
        fused_visit = self.fusion_layer(concatenated)
        return fused_visit

class LongitudinalPatientMemory(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        #
        self.modality_fusion = VisitModalityFusion(embed_dim=embed_dim)
        self.time_encoder = ContinuousTimeEncoder(embed_dim=embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, vision_feat, tabular_feat, text_feat, delta_t, time_mask):
        """
        time_mask parameter: boolean tensor (comes from dataloader).
        Where there is no visit (padding), its value is 0 or False.
        """
        #1Integrating different aspects of each visit
        fused_visits = self.modality_fusion(vision_feat, tabular_feat, text_feat)
        
        #2.Encode time and add it to visit vector
        time_encoded = self.time_encoder(delta_t)
        visits_with_time = fused_visits + time_encoded
        
        # 3. Convert time_mask to a format that TransformerEncoder understands
        #In PyTorch, for src_key_padding_mask, True values mean "Ignored"
        #dataloader would return 1 for the actual visit and 0 for padding.
        #So we need to invert it.
        padding_mask = ~time_mask 
        
        #4. Passing through the time series transformer
        longitudinal_features = self.temporal_transformer(
            visits_with_time, 
            src_key_padding_mask=padding_mask
        )
        
        return longitudinal_features

#Sanity Check
if __name__ == "__main__":
    B, T, D = 4, 5, 768
    
    # Simulate the output of encoders in phase 1
    dummy_vision = torch.randn(B, T, D)
    dummy_tabular = torch.randn(B, T, D)
    dummy_text = torch.randn(B, T, D)
    
    #Time simulation (e.g., days since the first visit)
    dummy_delta_t = torch.tensor([
        [0.0, 12.5, 45.0, 0.0, 0.0], # Patient 1: Three real visits, 2 padded.
        [0.0, 300.2, 0.0, 0.0, 0.0], #Patient 2: Two actual visits
        [0.0, 2.1, 5.0, 10.5, 20.0], #Patient 3: Five complete visits
        [0.0, 0.0, 0.0, 0.0, 0.0]    #Patient 4: One visit
    ])
    
    # Mask simulation (1=real, 0=padding)
    dummy_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0]
    ], dtype=torch.bool)
    
    #Model making
    memory_module = LongitudinalPatientMemory(embed_dim=768)
    
    #data transfer
    patient_state = memory_module(dummy_vision, dummy_tabular, dummy_text, dummy_delta_t, dummy_mask)
    
    print(f"Output Shape: {patient_state.shape}") 
    #  (4, 5, 768)