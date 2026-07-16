import torch
import torch.nn as nn

class ClinicalTabularEncoder(nn.Module):
    """
    Tabular data encoder (vital signs and triage).
    This module receives 6-dimensional numeric vectors and, after normalization,
    converts them into an Embedding Space shared with other facets.
    """
    def __init__(self, input_dim=6, embed_dim=768, hidden_dim=256, dropout_prob=0.5): #dropout_prob=0.1
        super().__init__()
        
        # 1.initial normalization layer:
        #to scale features with different statistical distributions
        self.norm_in = nn.LayerNorm(input_dim)
        
        #2.MLP for feature mapping
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),                # Modern activation function
            nn.Dropout(dropout_prob), #Avoiding Overfitting on Noise in Monitoring Devices
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)   #Final normalization for alignment with transformers
        )

    def forward(self, clinical_features):
        """:
        input 
            clinical_features: (B, T, input_dim) 
                 
        output:
          (B, T, embed_dim)
        """
        #In PyTorch, Linear layers inherently process the last dimension of the tensor.
        #Therefore, the time dimension (T) is handled automatically without the need for flattening.
        
        #Initial normalization on 6 features
        x = self.norm_in(clinical_features)
        
        #Passing through the neural network
        encoded_tabular = self.mlp(x)
        
        return encoded_tabular


#(Sanity Check)
if __name__ == "__main__":
    #Simulate a Batch of DataLoader Outputs: 4 Patients, 5 Visits Each, 6 Clinical Features
    dummy_clinical = torch.randn(4, 5, 6)
    
    # create model
    tabular_encoder = ClinicalTabularEncoder(input_dim=6, embed_dim=768)
    
    # passing
    output_features = tabular_encoder(dummy_clinical)
    
    print(f"Input Shape: {dummy_clinical.shape}")
    print(f"Output Features Shape: {output_features.shape}") 
    # (4, 5, 768)