import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses


class NtXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.ntxent_loss_fn = losses.NTXentLoss(temperature=temperature)

    def forward(self, ref_emb, query_emb):

        labels = torch.arange(ref_emb.size(0), device=ref_emb.device)
        labels = torch.cat([labels, labels], dim=0)
        embeddings = torch.cat([ref_emb, query_emb], dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        loss = self.ntxent_loss_fn(embeddings, labels)

        # remove labels from GPU
        labels.cpu()
        return loss
  

class TripletLoss(nn.Module):
    def __init__(self, margin=.5):
        super().__init__()
        self.triplet_miner = miners.TripletMarginMiner(margin=margin, type_of_triplets='semihard')
        self.triplet_loss_fn = losses.TripletMarginLoss(margin=margin, swap=False)
        
    def forward(self, ref_emb, query_emb):

        labels = torch.arange(ref_emb.size(0), device=ref_emb.device)
        labels = torch.cat([labels, labels], dim=0)
        embeddings = torch.cat([ref_emb, query_emb], dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        triplets = self.triplet_miner(embeddings, labels)
        loss = self.triplet_loss_fn(embeddings, labels, triplets)

        # remove labels from GPU
        labels.cpu()

        return loss


class TripletLossSupervised(nn.Module):
    def __init__(self, margin=.5):
        super().__init__()
        self.triplet_miner = miners.TripletMarginMiner(margin=margin, type_of_triplets='semihard')
        self.triplet_loss_fn = losses.TripletMarginLoss(margin=margin, swap=False)
        
    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)

        triplets = self.triplet_miner(embeddings, labels)
        loss = self.triplet_loss_fn(embeddings, labels, triplets)

        # remove labels from GPU
        labels.cpu()

        return loss


class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()        
    def forward(self, logits):

        labels = torch.arange(logits.size(0), device=logits.device).long()
        cross_loss_ref = F.cross_entropy(logits, labels)
        cross_loss_query = F.cross_entropy(logits.T, labels)
        loss = (cross_loss_ref+cross_loss_query)/2

        # remove labels from GPU
        labels.cpu()

        return loss


class TripletEntropyLoss(nn.Module):

    def __init__(self, cel_weigth_tensor=None, margin=0.5, triplet_type='semihard', cel_weigth=1, te_weight=1):
        super().__init__()

        self.margin = margin
        self.cel_weight = cel_weigth
        self.te_weight = te_weight
        self.triplet_type = triplet_type
        self.triplet_miner = miners.TripletMarginMiner(margin=self.margin, type_of_triplets=self.triplet_type)
        self.triplet_loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=False)

        if not cel_weigth_tensor is None:
            self.cel_weight_tensor = cel_weigth_tensor
            self.cross_entropy_loss_fn = nn.CrossEntropyLoss(weight=cel_weigth_tensor)
        else:
            self.cross_entropy_loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, embeddings, labels):

        # calculate triplet loss
        embeddings = F.normalize(embeddings, p=2, dim=1)

        triplets = self.triplet_miner(embeddings, labels)
        triplet_loss = self.triplet_loss_fn(embeddings, labels, triplets)


        #calculate cross entropy loss
        cross_entropy_loss = self.cross_entropy_loss_fn(logits, labels)
        loss = self.cel_weight * cross_entropy_loss + self.te_weight * triplet_loss
        return loss

