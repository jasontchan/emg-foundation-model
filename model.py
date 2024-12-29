import torch
import torch.nn as nn
import torch.nn.functional as F

from hash_embedding import HashEmbedding
from perceiver_rotary import PerceiverRotary
from utilities import create_output_queries
from infinite_embedding_new import InfiniteVocabEmbedding


class Model(nn.Module):
    """
    in the init function, create embedding objects and perceiverIO object

    in the forward function, actually embed the data and run through perceiverIO while outputing the loss etc
    """

    def __init__(
        self,
        embedding_dim, #model dimension
        session_emb_dim,
        subject_emb_dim,
        num_latents,
        latent_dim,
        num_classes,
        dropout=0.1,
    ):
        super().__init__()

        # store model parameters
        self.num_latents = num_latents
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # self.input_embedding = InfiniteVocabEmbedding(embedding_dim=embedding_dim)
        # self.input_embedding.load_state_dict(torch.load('data/infinite_vocab_embedding.pt'))
        # self.input_embedding.extend_vocab(torch.tensor([0.0, 0.0, 0.0, 0.0]))


        self.session_embedding = InfiniteVocabEmbedding(embedding_dim=session_emb_dim)
        self.session_embedding.load_state_dict(torch.load('data/session_vocab_embedding.pt'))
        self.session_embedding.extend_vocab("0") #dk if this is necessary
        print(self.session_embedding.vocab)
        

        self.subject_embedding = InfiniteVocabEmbedding(embedding_dim=subject_emb_dim)
        self.subject_embedding.load_state_dict(torch.load('data/subject_vocab_embedding.pt'))
        self.subject_embedding.extend_vocab("0") #dk if this is necessary

        self.channel_embedding = nn.Embedding(17, embedding_dim=8)
        self.latent_embedding = nn.Embedding(num_latents, embedding_dim=latent_dim)

        inner_dimension = session_emb_dim + subject_emb_dim + 8 + 2
        self.projection = nn.Linear(inner_dimension, embedding_dim)

        self.dropout = nn.Dropout(dropout)

        # create perceiverIO object
        self.perceiver_io = PerceiverRotary(
            dim=embedding_dim,
            context_dim=embedding_dim,
            dim_head=64,
            depth=2,
            cross_heads=1,
            self_heads=8,
            ffn_dropout=dropout,
            lin_dropout=dropout,
            atn_dropout=dropout,
        )

        # (A) Learned classification query (shape: [dim])
        self.class_query = nn.Parameter(torch.randn(embedding_dim))

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.readout = nn.Linear(embedding_dim, self.num_classes)  # predictions + loss
        self.dim = embedding_dim  # double check d

    def create_padding_mask(  # TODO: consider moving this to utilities?
        self, sequence_lengths: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        batch_size = sequence_lengths.size(0)
        mask = (
            torch.arange(max_length, device=sequence_lengths.device)[None, :]
            >= sequence_lengths[:, None]
        )
        return mask  # shape: (batch_size, max_length)

    def forward(
        self,
        data,
        sequence_lengths,
        time_stamps,
        latent_idx,
        latent_timestamps,
        labels=None,
        # output_batch_index
    ):
        sessions = data[:, :, 0]      # [B, seq_len]
        session_ids = torch.tensor([self.session_embedding.tokenizer(str(int(session.item()))) for session_seq in sessions for session in session_seq]) #TODO: check this
        subjects = data[:, :, 1]      # [B, seq_len]
        subject_ids = torch.tensor([self.subject_embedding.tokenizer(str(int(subject.item()))) for subject_seq in subjects for subject in subject_seq]) #TODO: check this
        channel_ids = data[:, :, 2]      # [B, seq_len] channels and channel ids are the same!
        prominence  = data[:, :, 3]     # [B, seq_len]
        duration    = data[:, :, 4]     # [B, seq_len]

        batch_size = data.size(0)
        max_seq_len = data.size(1)
        padding_mask = self.create_padding_mask(
            sequence_lengths=sequence_lengths, max_length=max_seq_len
        )
        # indices = torch.tensor([[self.input_embedding.tokenizer(sequence)] for sequence in data])
        # print("indices", indices)

        session_emb = self.session_embedding(session_ids)
        subject_emb = self.subject_embedding(subject_ids)
        channel_emb = self.channel_embedding(channel_ids.long())
        #shape prom and dur so can concat
        prominence = prominence.unsqueeze(-1)
        duration = duration.unsqueeze(-1)

        #COMBINE EVERYTHING SO THEYRE ALL TOGETHER
        inputs = torch.cat([session_emb, subject_emb, channel_emb, prominence, duration], dim=-1)
        print("COMBINED INPUTS", inputs)
        raise KeyboardInterrupt
        inputs = self.projection(inputs)
        # inputs = self.input_embedding(indices)  # (batch_size, 1(?), max_seq_len, embedding_dim)
        # inputs = inputs.squeeze(1) #get rid of singleton dimension so its size [batch_size, max_seq_len, embedding_dim]
        inputs = self.dropout(inputs)

        latents = self.latent_embedding(latent_idx) #size [n_latents, lat_emb_dim]
        latents = latents.unsqueeze(0).expand(batch_size, -1, -1) #size [batch_size, n_latents, lat_emb_dim]
        latent_timestamps = latent_timestamps.unsqueeze(0).expand(batch_size, -1)
        # # create output queries
        # output_timestamps, output_queries = create_output_queries(
        #     1.0, 1, batch_size, self.embedding_dim #max time 1.0, n_output queries = 1 (classification), b, dim
        # )
        output_queries = self.class_query.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1)
        output_timestamps = torch.tensor([1.0 for _ in range(batch_size)])
        output_timestamps = output_timestamps.unsqueeze(1)


        # run through perceiverIO and return loss
        output_latents = self.perceiver_io(
            inputs=inputs,
            latents=latents,
            output_queries=output_queries,  # how do we omit this for training? do we need to omit for training?
            input_timestamps=time_stamps,
            latent_timestamps=latent_timestamps,
            output_query_timestamps=output_timestamps,
            input_mask=padding_mask,
            input_seqlen=sequence_lengths,
            latent_seqlen=torch.full_like(sequence_lengths, self.num_latents),
            output_query_seqlen=torch.ones_like(sequence_lengths),
        )
        output_latents = self.layer_norm(output_latents)
        predictions = self.readout(output_latents).squeeze(1)
        print("PREDICTIONS", predictions)
        if torch.isnan(predictions).any():
            print("NaN detected in predictions")
            print("Output latents stats:", 
                "mean:", output_latents.mean().item(),
                "std:", output_latents.std().item(),
                "min:", output_latents.min().item(),
                "max:", output_latents.max().item())
        loss = None
        if labels is not None:
            print("LABELS", labels)
            loss = F.cross_entropy(predictions, labels.long())

        return predictions, loss

    def predict(
        self,
        data: torch.Tensor,
        sequence_lengths: torch.Tensor,
        time_stamps: torch.Tensor,
        latent_timestamps: torch.Tensor,
        output_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prediction method for inference.
        Returns class probabilities after softmax.
        """
        predictions, _ = self.forward(
            data=data,
            sequence_lengths=sequence_lengths,
            time_stamps=time_stamps,
            latent_timestamps=latent_timestamps,
            output_timestamps=output_timestamps,
        )
        return F.softmax(predictions, dim=-1)

    # @staticmethod
    # def collate_fn(batch): #NOTE: why do i have another collate function here ....? looks like its not being used ok
    #     """
    #     Custom collate function for handling variable length sequences in DataLoader.

    #     Args:
    #         batch: List of tensors with variable lengths

    #     Returns:
    #         Padded batch tensor and sequence lengths
    #     """
    #     # Sort batch by sequence length (descending)
    #     batch.sort(key=lambda x: x.size(0), reverse=True)

    #     # Get sequence lengths
    #     lengths = torch.tensor([x.size(0) for x in batch])

    #     # Pad sequences
    #     padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

    #     return padded, lengths
