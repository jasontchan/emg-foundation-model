import torch
import torch.nn as nn

#lowkey gonna use infinite instead of hash rn ok bye
#why do we use hash embedding instead of normal embedding again?
class HashEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_buckets=None, seed=0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets or num_embeddings
        self.seed = seed
        self.embeddings = nn.Embedding(self.num_buckets, embedding_dim)

    def forward(self, input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        hashed_ids = self._hash(input_ids)
        return self.embeddings(hashed_ids)

    def _hash(self, input_ids):
        return (input_ids * 2654435761 + self.seed) % self.num_buckets