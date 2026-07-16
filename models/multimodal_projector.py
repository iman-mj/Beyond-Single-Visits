import torch
import torch.nn as nn

class MultimodalProjector(nn.Module):
    """
        Multifaceted Projector.
        This module translates the Personalized Patient State Vectors into the Embedding Space of the large language model.
    """
    def __init__(self, input_dim=768, llm_dim=4096, hidden_dim=2048, num_tokens=16, dropout=0.5): #dropout=0.2
        super().__init__()
        #----------------------------
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        #----------------------------
        # Using a two-layer MLP instead of a simple linear layer.
        # This greatly increases the model's capacity to align complex medical concepts with natural language tokens.
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens * llm_dim)
        )
        # self.projector = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, llm_dim)
        # )

    def forward(self, patient_state_vector):
        """
        Input:
        patient_state_vector: Output of phase 2 with dimensions (B, T, 768)
        Output:
        Tensor with dimensions (B, T, 4096) which is directly given as
        "Soft Tokens" as a prefix to the LLM.
        """
        # In PyTorch, linear layers are automatically applied only to the last dimension (Feature).
        # So the time dimension (T) remains untouched and the network produces a separate vector for each visit
        # for the LLM.
        #----------------------------
        x = self.projector(patient_state_vector)
        llm_embeddings = x.view(-1, self.num_tokens, self.llm_dim)
        #----------------------------
        #llm_embeddings = self.projector(patient_state_vector)
        
        return llm_embeddings

#(Sanity Check)
if __name__ == "__main__":
    B, T, input_D = 4, 5, 768
    
    # Let's assume this is the output of the Longitudinal Patient Memory module
    dummy_patient_states = torch.randn(B, T, input_D)
    
    # Build a projector to connect to Llama-3 8B (whose input dimensions are 4096)
    projector = MultimodalProjector(input_dim=768, llm_dim=4096)
    
    # Data Passage
    llm_ready_tokens = projector(dummy_patient_states)
    
    print(f"Input Shape (From Phase 2): {dummy_patient_states.shape}")
    print(f"Output Shape (Ready for LLM): {llm_ready_tokens.shape}") 
    #The output will be (4, 5, 4096).
    #Now LLM sees this tensor as 5 very brainy words (tokens) containing clinical history