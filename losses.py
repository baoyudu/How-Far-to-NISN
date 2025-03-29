import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, eeg_embeddings, text_embeddings):
        eeg_embeddings = F.normalize(eeg_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        
        similarity = torch.matmul(eeg_embeddings, text_embeddings.T) / self.temperature
        
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        
        loss = F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.T, labels)
        return loss / 2

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, eeg_embeddings, text_embeddings, categories):
        zero_loss = torch.tensor(0.0, device=eeg_embeddings.device, requires_grad=True)
        
        loss = zero_loss
        N = eeg_embeddings.size(0)
        
        if len(categories.unique()) < 2:
            return zero_loss
        
        valid_triplets = 0
        
        for i in range(N):
            pos_indices = (categories == categories[i]).nonzero().squeeze()
            neg_indices = (categories != categories[i]).nonzero().squeeze()
            
            if pos_indices.dim() == 0 or neg_indices.dim() == 0:
                continue
                
            pos_candidates = pos_indices[pos_indices != i]
            if len(pos_candidates) == 0:
                continue
                
            pos_idx = pos_candidates[torch.randint(len(pos_candidates), (1,))]
            neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))]
            
            anchor = eeg_embeddings[i]
            positive = text_embeddings[pos_idx]
            negative = text_embeddings[neg_idx]
            
            pos_dist = F.pairwise_distance(anchor.unsqueeze(0), positive.unsqueeze(0))
            neg_dist = F.pairwise_distance(anchor.unsqueeze(0), negative.unsqueeze(0))
            
            loss = loss + F.relu(pos_dist - neg_dist + self.margin)
            valid_triplets += 1
            
        return loss / valid_triplets if valid_triplets > 0 else zero_loss 