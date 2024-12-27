from infinite_embedding_test import InfiniteVocabEmbedding
import torch
from utilities import create_output_queries
    
# embedding = InfiniteVocabEmbedding(embedding_dim=256)
# embedding.load_state_dict(torch.load('data/infinite_vocab_embedding.pt'))

# # print(embedding.tokenizer(torch.tensor([99999.0, 4, 2.8, .02])))

# print(len(embedding.vocab))

# for key, val in embedding.vocab.items():
#     if val < 1000:
#         print(key, val)
# # embedding.extend_vocab()
# # print(embedding.vocab['0 0 0 0'])

output_timestamps, output_queries = create_output_queries(1.0, 1, 3, 256)
print(output_timestamps)
print(output_queries)
print(output_queries.size())