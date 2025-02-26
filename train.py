import gzip
import os
import pickle
import h5py
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import create_linspace_latent_tokens, create_output_queries
from model import Model
from spike_dataset import SpikeDataset
from infinite_embedding_test import InfiniteVocabEmbedding

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        gesture_names,
        learning_rate=1e-4,
        weight_decay=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # device="cpu",
    ):

        # set member variables
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gesture_names = gesture_names
        self.device = device

        # initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )

        self.latent_idx, self.latent_timestamps = create_linspace_latent_tokens(
            0, 1.0, 0.125, 32
        )

    def train_epoch(self):
        self.model.train()

        running_loss = 0.0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(self.train_loader, desc="Training")

        for (
            sessions,
            subjects,
            channels,
            prominences,
            durations,
            timestamps,
            lengths,
            labels,
        ) in progress_bar:

            sessions = sessions.to(self.device)
            subjects = subjects.to(self.device)
            channels = channels.to(self.device)
            prominences = prominences.to(self.device)
            durations = durations.to(self.device)
            timestamps = timestamps.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            features = torch.stack(
                (sessions, subjects, channels, prominences, durations), dim=1
            )
            features = torch.transpose(features, 1, 2)
            # zero gradients
            self.optimizer.zero_grad()

            # forward pass!
            with torch.cuda.amp.autocast(enabled=False):
                predictions, loss = self.model(
                    data=features,
                    sequence_lengths=lengths,
                    time_stamps=timestamps,
                    latent_timestamps=self.latent_timestamps,
                    latent_idx=self.latent_idx,
                    labels=labels,
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()
            pred_labels = torch.argmax(predictions, dim=1)
            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

        return {"loss": epoch_loss, "accuracy": epoch_acc, "f1": epoch_f1}

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        running_loss = 0.0
        all_preds = []
        all_labels = []

        for (
            sessions,
            subjects,
            channels,
            prominences,
            durations,
            timestamps,
            lengths,
            labels,
        ) in self.val_loader:

            sessions = sessions.to(self.device)
            subjects = subjects.to(self.device)
            channels = channels.to(self.device)
            prominences = prominences.to(self.device)
            durations = durations.to(self.device)
            timestamps = timestamps.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            features = torch.stack(
                (sessions, subjects, channels, prominences, durations), dim=1
            )
            features = torch.transpose(features, 1, 2)

            # forward pass!
            predictions, loss = self.model(
                data=features,
                sequence_lengths=lengths,
                time_stamps=timestamps,
                latent_timestamps=self.latent_timestamps,
                latent_idx=self.latent_idx,
                labels=labels,
            )

            running_loss += loss.item()
            pred_labels = torch.argmax(predictions, dim=1)
            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")

        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)

        return {"loss": val_loss, "accuracy": val_acc, "f1": val_f1}

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.gesture_names,
            yticklabels=self.gesture_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig("confusion_mat.png")
        plt.close()

    def train(self, n_epochs):

        for epoch in range(n_epochs):
            print("epoch number:", epoch)

            train_metrics = self.train_epoch()

            val_metrics = self.validate()

            metrics = {
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "epoch": epoch,
            }

            print(
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}"
            )
            print(
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}"
            )

            # Learning rate scheduling
            self.scheduler.step(val_metrics["f1"])

            # TODO: potentially save the model

    def predict(self):
        pass


if __name__ == "__main__":
    embedding_dim = 256
    #need session_idx, stage_idx (key idx), subject_idx, and data
    DATA_STORE = "full_data/"
    DATA_BATCH_SIZE = 50
    NUM_TRAIN_FILES = 200
    NUM_VAL_FILES = 50
    if os.path.exists(DATA_STORE + "key_idx.pickle.gz"):
        with gzip.open(DATA_STORE + "key_idx.pickle.gz", "rb") as f:
            stage_idx = pickle.load(f)
    if os.path.exists(DATA_STORE + "session_idx.pickle.gz"):
        with gzip.open(DATA_STORE + "session_idx.pickle.gz", "rb") as f:
            session_idx = pickle.load(f)
    if os.path.exists(DATA_STORE + "subject_idx.pickle.gz"):
        with gzip.open(DATA_STORE + "subject_idx.pickle.gz", "rb") as f:
            subject_idx = pickle.load(f)
    
    #load train data
    train_data = torch.tensor([])
    for i in range(0, NUM_TRAIN_FILES, DATA_BATCH_SIZE):
        with h5py.File(DATA_STORE + "train_data_batch_"+str(i)+".h5", "r") as hdf5_file:
            train_batch = list(hdf5_file.keys())[0]
            batch = torch.tensor(hdf5_file[train_batch][:])
            train_data = torch.cat((train_data, batch), dim=0)

    #load val data
    val_data = torch.tensor([])
    for i in range(0, NUM_VAL_FILES, DATA_BATCH_SIZE):
        with h5py.File(DATA_STORE + "validation_data_batch_"+str(i)+".h5", "r") as hdf5_file:
            val_batch = list(hdf5_file.keys())[0]
            batch = torch.tensor(hdf5_file[val_batch][:])
            val_data = torch.cat((val_data, batch), dim=0)
    
    #NOTE: this is a one-time use case bc val has some subjects who arent in train
    val_data = val_data[val_data[:, 1] <= 83]

    train_indices = list(range(len(train_data)))
    val_indices = list(range(len(val_data)))
    labels = set(stage_idx.values())
    print("STAGE IDX", stage_idx)

    # print("training_data", training_data)
    train_spike_token_data = SpikeDataset(train_data)
    val_spike_token_data = SpikeDataset(val_data)

    # sample_item = train_spike_token_data[0]
    # print("Sample item from dataset:")
    # print("Type:", type(sample_item))
    # print("Length:", len(sample_item))
    # print("Contents:", sample_item)

    # print("amount of data in training", len(train_spike_token_data))
    train_loader = DataLoader(
        train_spike_token_data,
        batch_size=32,  
        shuffle=True,
        collate_fn=SpikeDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_spike_token_data,
        batch_size=32,
        shuffle=True,
        collate_fn=SpikeDataset.collate_fn,
    )
    # need parameters 'num_embeddings', 'embedding_dim', 'num_buckets', 'num_latents', and 'latent_dim'
    model = Model(
        embedding_dim=embedding_dim,
        session_emb_dim=8,
        subject_emb_dim=8,
        num_latents=256,
        latent_dim=256,
        num_classes=len(list(stage_idx.values())),
    ).to(device)
    model.double()
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        gesture_names=list(stage_idx.values()),
        learning_rate=1e-4,
        weight_decay=0.01,
    )

    # Train model
    trainer.train(n_epochs=20)

    # Make predictions
    # test_predictions, test_confidences = trainer.predict(_loader)

    # Print some predictions with their confidence
    # for pred, conf in zip(test_predictions[:5], test_confidences[:5]):
    #     print(
    #         f"Predicted gesture: {trainer.gesture_names[pred]} with confidence: {conf:.4f}"
    #     )
