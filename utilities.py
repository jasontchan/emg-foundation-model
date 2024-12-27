import numpy as np
from einops import repeat
import torch


def create_linspace_latent_tokens(start, end, step, num_latents_per_step):
    r"""Creates a sequence of latent tokens. Each token is defined by the
    latent index and the timestamps. The sequence is defined by the start and end
    time and the step size. The group of `num_latents_per_step` latents is repeated
    for each step.

    Args:
        start (float): The start time of the sequence.
        end (float): The end time of the sequence.
        step (float): The step size.
        num_latents_per_step (int): The number of latents per step.
    """
    sequence_len = end - start
    latent_timestamps = np.arange(0, sequence_len, step) + step / 2 + start
    latent_index = np.arange(num_latents_per_step, dtype=np.int64)

    num_timestamps = len(latent_timestamps)
    latent_timestamps = repeat(latent_timestamps, "t -> (t u)", u=len(latent_index))

    latent_index = repeat(latent_index, "u -> (t u)", t=num_timestamps)

    return torch.tensor(latent_index, dtype=torch.long), torch.tensor(
        latent_timestamps, dtype=torch.float32
    )


def create_output_queries(max_time, num_queries, batch_size, embedding_dim):
    timestamps = torch.linspace(0, max_time, num_queries)
    timestamps = timestamps.unsqueeze(0).repeat(batch_size, 1)
    output_queries = torch.zeros(
        (batch_size, num_queries, embedding_dim),
        # device=inputs.device,
        # dtype=inputs.dtype
    )
    return timestamps.clone().detach(), output_queries.clone().detach()


if __name__ == "__main__":
    latent_idx, latent_tstamps = create_linspace_latent_tokens(0, 3.0, 0.375, 32)
    print(latent_idx, latent_tstamps)
