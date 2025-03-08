import torch
import torch.nn as nn
import torch.nn.functional as F

from hash_embedding import HashEmbedding
from perceiver_rotary import PerceiverRotary
from utilities import create_output_queries
from infinite_embedding_new import InfiniteVocabEmbedding
from deepspeed.runtime.zero.partition_parameters import GatheredParameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    """
    main model
    """

    def __init__(
        self,
        embedding_dim,  # model dimension
        session_emb_dim,
        subject_emb_dim,
        num_latents,
        latent_dim,
        num_classes,
        emb_directory,
        dropout=0.1,
        device=device,
    ):
        super().__init__()

        self.num_latents = num_latents
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.num_channels = 32
        self.channel_emb_dim = 16

        self.device = device

        # embed session
        self.session_embedding = InfiniteVocabEmbedding(embedding_dim=session_emb_dim).to(device)
        self.session_embedding.load_state_dict(
            torch.load(emb_directory + "session_vocab_embedding.pt")
        )
        self.session_embedding.extend_vocab("0")  # dk if this is necessary

        # embed subject
        self.subject_embedding = InfiniteVocabEmbedding(embedding_dim=subject_emb_dim).to(device)
        self.subject_embedding.load_state_dict(
            torch.load(emb_directory + "subject_vocab_embedding.pt")
        )
        self.subject_embedding.extend_vocab("0")  # dk if this is necessary

        self.channel_embedding = nn.Embedding(self.num_channels+1, embedding_dim=self.channel_emb_dim).to(device)
        self.latent_embedding = nn.Embedding(num_latents, embedding_dim=latent_dim).to(device)

        inner_dimension = session_emb_dim + subject_emb_dim + self.channel_emb_dim + 2
        self.projection = nn.Linear(inner_dimension, embedding_dim)

        self.dropout = nn.Dropout(dropout)

        self.perceiver_io = PerceiverRotary(
            dim=embedding_dim,
            context_dim=embedding_dim,
            dim_head=64,
            depth=2, #was 2
            cross_heads=1,
            self_heads=8,
            ffn_dropout=dropout,
            lin_dropout=dropout,
            atn_dropout=dropout,
        )

        self.class_query = nn.Parameter(torch.randn(embedding_dim))

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.readout = nn.Linear(
            embedding_dim, self.num_classes,
        )  # predictions + loss
        self.dim = embedding_dim  # double check d

    def create_padding_mask(
        self, sequence_lengths: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        batch_size = sequence_lengths.size(0)
        mask = (
            torch.arange(max_length, device=sequence_lengths.device)[None, :]
            < sequence_lengths[:, None]
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
        sessions = data[:, :, 0]  # [B, seq_len]
        # print("SESSIONS", sessions)
        # print("max session", sessions.max())
        session_ids = torch.tensor(
            [
                [
                    self.session_embedding.tokenizer(str(int(session.item())))
                    for session in session_seq
                ]
                for session_seq in sessions
            ]
        )  # TODO: check this
        # print("SESSION IDS", session_ids)
        # print("session ids shape", session_ids.shape)
        # print("Max session index:", torch.flatten(session_ids).max().item())  

        session_ids = session_ids.to(self.device)
        subjects = data[:, :, 1]  # [B, seq_len]
        subject_ids = torch.tensor(
            [
                [
                    self.subject_embedding.tokenizer(str(int(subject.item())))
                    for subject in subject_seq
                ]
                for subject_seq in subjects
            ]
        )  # TODO: check this
        # print("SUBJECT IDs", subject_ids)
        # print("SUBJECT IDs shape", subject_ids.shape)
        subject_ids = subject_ids.to(self.device)
        channel_ids = data[
            :, :, 2
        ]  # [B, seq_len] channels and channel ids are the same!
        prominence = data[:, :, 3]  # [B, seq_len]
        duration = data[:, :, 4]  # [B, seq_len]

        batch_size = data.size(0)
        max_seq_len = data.size(1)
        padding_mask = self.create_padding_mask(
            sequence_lengths=sequence_lengths, max_length=max_seq_len
        )
        # print("padding_mask", padding_mask)
        # print("padding mask size", padding_mask.size())

        session_emb = self.session_embedding(session_ids).to(self.device)
        subject_emb = self.subject_embedding(subject_ids).to(self.device)
        channel_emb = self.channel_embedding(channel_ids.long()).to(self.device)

        # unsqueeze prom and dur so can concat
        prominence = prominence.unsqueeze(-1)
        duration = duration.unsqueeze(-1)

        # print("PROMINENCE", prominence)
        # print("DURATION", duration)
        session_emb = session_emb.to(self.device)
        subject_emb = subject_emb.to(self.device)
        channel_emb = channel_emb.to(self.device)
        prominence = prominence.to(self.device)
        duration = duration.to(self.device)

        # COMBINE EVERYTHING SO THEYRE ALL TOGETHER
        inputs = torch.cat(
            [session_emb, subject_emb, channel_emb, prominence, duration], dim=-1
        ).to(self.device)
        # print("INPUTSHERHE", inputs)
        # print("INPUTSHERHEH shape", inputs.shape)
        # inputs = inputs.to(torch.float32)
        # print("INPUTS AFTER TO TORCH 32", inputs)
        inputs = self.projection(inputs)
        # print("INPUTS AFTER PROJECTION", inputs)
        # print("INPUTS AFTER PROJECTION SHAPE", inputs.shape)
        inputs = self.dropout(inputs)
        # print("INPUTS", inputs)

        latent_idx = latent_idx.to(self.device)
        latents = self.latent_embedding(latent_idx)  # size [n_latents, lat_emb_dim]
        latents = latents.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # size [batch_size, n_latents, lat_emb_dim]
        latent_timestamps = latent_timestamps.unsqueeze(0).expand(batch_size, -1)
        # print("latent_timestamps", latent_timestamps)
        # # create output queries
        # output_timestamps, output_queries = create_output_queries(
        #     1.0, 1, batch_size, self.embedding_dim #max time 1.0, n_output queries = 1 (classification), b, dim
        # )
        output_queries = (
            self.class_query.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1)
        )
        output_timestamps = torch.tensor([1.0 for _ in range(batch_size)]).to(self.device)
        output_timestamps = output_timestamps.unsqueeze(1)
        # print("OUTPUT QUERIES", output_queries)

        # run through perceiverIO and return loss
        output_latents = self.perceiver_io(
            inputs=inputs,
            latents=latents,
            output_queries=output_queries,
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
        # print("PREDICTIONS", predictions)
        loss = None
        if labels is not None:
            # print("LABELS", labels)
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
