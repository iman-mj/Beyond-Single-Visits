import torch
import torch.nn as nn
import torchvision.models as models

class AttentionPooling(nn.Module):
    """
    Smart Microscope (Attention Pooling)
    Instead of simple averaging,
     a query vector learns which parts of the image (which of the 49 patches)
      to pay more attention to.
    """
    def __init__(self, in_dim=1024, out_dim=768, num_heads=8):
        super().__init__()
        # # A learning vector
        self.query = nn.Parameter(torch.randn(1, 1, in_dim))
        
        #Attention layer to compare the query with all image patches
        self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(in_dim)
        
        ## The ultimate projector to 768-dimensional space
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x shape from Swin: (B, C, H, W) e.g., (Batch, 1024, 7, 7)
        B, C, H, W = x.shape
        
        #Converting a 2D map into a sequence of patches (49 patches)
        # shape: (B, H*W, C) -> (Batch, 49, 1024)
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        
        # Copying Query for All Images in Batch
        # shape: (B, 1, 1024)
        q = self.query.expand(B, -1, -1)
        
        #Attention calculation: Our query looks at 49 patches and only extracts the important information.
        # The output attn_out is a vector that is an intelligent extract of the entire image.
        attn_out, attn_weights = self.attention(q, x_flat, x_flat)
        
        # Residual connection + Norm
        out = self.norm(q + attn_out).squeeze(1) # shape: (B, 1024)
        
        # Convert to 768 space
        return self.proj(out) # shape: (B, 768)


class CXRSwinExtractor(nn.Module):
    """
    Feature extractor from single image using modern Swin Transformer network
    with Advanced Attention Pooling.
    """
    def __init__(self, embed_dim=768, freeze_base=True):
        super().__init__()
        
        #sing Swin Transformer V2 (Base Model)
        swin = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
        
        #extracting feature layers (without final classification layer)
        self.features = swin.features
        self.norm = swin.norm
        self.permute = swin.permute
        
        # freezing early transformer layers to optimize VRAM memory
        if freeze_base:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.norm.parameters():
                param.requires_grad = False
                
        #Attention Pooling
        self.attn_pool = AttentionPooling(in_dim=1024, out_dim=embed_dim)

    def forward(self, x):
        # passing the image through Swin Transformer blocks
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)

        #attention polling
        x = self.attn_pool(x)
        
        return x

class CrossViewAttention(nn.Module):
    """
    Attention module for intelligently combining front and side views.
    """
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, frontal_feat, lateral_feat):
        seq = torch.stack([frontal_feat, lateral_feat], dim=1)
        
        attn_out, _ = self.attention(seq, seq, seq)
        seq = self.norm1(seq + attn_out)
        
        ffn_out = self.ffn(seq)
        seq = self.norm2(seq + ffn_out)
        
        fused_feat = seq.mean(dim=1)
        return fused_feat

class MultiViewVisionEncoder(nn.Module):
    """
    The final wrapper class that connects directly to your DataLoader output.
    """
    def __init__(self, embed_dim=768, freeze_base=True):
        super().__init__()
        self.extractor = CXRSwinExtractor(embed_dim=embed_dim, freeze_base=freeze_base)
        self.fusion_module = CrossViewAttention(embed_dim=embed_dim)

    def forward(self, frontal_imgs, lateral_imgs):
        B, T, C, H, W = frontal_imgs.shape
        
        f_imgs_flat = frontal_imgs.view(B * T, C, H, W)
        l_imgs_flat = lateral_imgs.view(B * T, C, H, W)
        
        f_feats_flat = self.extractor(f_imgs_flat) 
        l_feats_flat = self.extractor(l_imgs_flat)
        
        fused_feats_flat = self.fusion_module(f_feats_flat, l_feats_flat)
        
        final_out = fused_feats_flat.view(B, T, -1)
        
        return final_out

#Sanity Chec
if __name__ == "__main__":
    dummy_frontal = torch.randn(4, 5, 3, 224, 224)
    dummy_lateral = torch.randn(4, 5, 3, 224, 224)
    
    vision_encoder = MultiViewVisionEncoder(embed_dim=768)
    
    output_features = vision_encoder(dummy_frontal, dummy_lateral)
    
    print(f"Input Shape: {dummy_frontal.shape}")
    print(f"Output Features Shape: {output_features.shape}") 
    # (4, 5, 768)