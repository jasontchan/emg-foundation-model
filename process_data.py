import pickle
import torch
from infinite_embedding_new import InfiniteVocabEmbedding

if __name__ == "__main__":
    embedding_dim = 256
    session_emb_dim = 8
    subject_emb_dim = 8

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

    # construct vocabulary!
    # infinite_vocab = InfiniteVocabEmbedding(embedding_dim=embedding_dim)
    # tokens = [spike[1:-3].clone().detach() for spike in input_tensor]
    # infinite_vocab.initialize_vocab(tokens)

    # torch.save(infinite_vocab.state_dict(), 'data/infinite_vocab_embedding.pt')

    session_vocab = InfiniteVocabEmbedding(embedding_dim=session_emb_dim)
    sessions = [str(int(spike[0])) for spike in input_tensor]
    session_vocab.initialize_vocab(sessions)

    torch.save(session_vocab.state_dict(), "data/session_vocab_embedding.pt")

    subject_vocab = InfiniteVocabEmbedding(embedding_dim=subject_emb_dim)
    subjects = [str(int(spike[1])) for spike in input_tensor]
    subject_vocab.initialize_vocab(subjects)

    torch.save(subject_vocab.state_dict(), "data/subject_vocab_embedding.pt")

