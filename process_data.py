import pickle
import torch
from infinite_embedding_test import InfiniteVocabEmbedding

if __name__ == "__main__":
    embedding_dim = 256

    with open("data/all_spikes.pickle", "rb") as file:
        all_spikes = pickle.load(file)
    # print(all_spikes)
    all_times = torch.tensor([dict["time"] for dict in all_spikes])
    all_sessions = torch.tensor([dict["session"] for dict in all_spikes])
    all_subjects = torch.tensor([dict["subject"] for dict in all_spikes])
    all_channels = torch.tensor([dict["channel"] for dict in all_spikes])
    all_prominences = torch.tensor([dict["prominence"] for dict in all_spikes])
    all_durations = torch.tensor([dict["duration"] for dict in all_spikes])
    all_gestures = torch.tensor([dict["gesture"] for dict in all_spikes])
    all_gesture_instances = torch.tensor([int(dict["instance"]) for dict in all_spikes])
    input_tensor = torch.vstack(
        (
            all_sessions,
            all_subjects,
            all_channels,
            all_prominences,
            all_durations,
            all_times,
            all_gesture_instances,
            all_gestures,
        )
    )
    input_tensor = input_tensor.t()
    with open("data/input_tensor.pickle", "wb") as handle:
        pickle.dump(input_tensor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #construct vocabulary!
    infinite_vocab = InfiniteVocabEmbedding(embedding_dim=embedding_dim)
    tokens = [spike[1:-3].clone().detach() for spike in input_tensor]
    infinite_vocab.initialize_vocab(tokens)

    torch.save(infinite_vocab.state_dict(), 'data/infinite_vocab_embedding.pt')