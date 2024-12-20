import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib as plt
import seaborn as sns

from create_latents import create_linspace_latent_tokens, create_output_queries
from model import Model
from spike_token_dataset import SpikeTokenDataset


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
            0, 3.0, 0.375, 32
        )

    def train_epoch(self):
        self.model.train()

        running_loss = 0.0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(self.train_loader, desc="Training")

        for features, timestamps, lengths, labels in progress_bar:

            features = features.to(self.device)
            timestamps = timestamps.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            # zero gradients
            self.optimizer.zero_grad()

            # forward pass!
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

        for features, timestamps, lengths, labels in self.val_loader:

            features = features.to(self.device)
            timestamps = timestamps.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            # forward pass!
            predictions, loss = self.model(
                data=features,
                sequence_lengths=lengths,
                time_stamps=timestamps,
                latent_timestamps=self.latent_timestamps,
                output_timestamps=None,
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

    with open("data_collection/data/spike_data/all_spikes.pickle", "rb") as file:
        all_spikes = pickle.load(file)

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

    data = input_tensor.t()
    # print("input tensor", data)

    indices = list(range(len(data)))
    # print("indices", indices)
    train_indices, val_indices = train_test_split(indices, test_size=0.2)
    # print("data", data)
    # print("train_indices", train_indices)
    training_data = data[train_indices]
    val_data = data[val_indices]

    # print("training_data", training_data)
    train_spike_token_data = SpikeTokenDataset(training_data)
    val_spike_token_data = SpikeTokenDataset(val_data)

    sample_item = train_spike_token_data[0]
    print("Sample item from dataset:")
    print("Type:", type(sample_item))
    print("Length:", len(sample_item))
    print("Contents:", sample_item)

    train_loader = DataLoader(
        train_spike_token_data,
        batch_size=64,
        shuffle=True,
        collate_fn=SpikeTokenDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_spike_token_data,
        batch_size=64,
        shuffle=True,
        collate_fn=SpikeTokenDataset.collate_fn,
    )

    # need parameters 'num_embeddings', 'embedding_dim', 'num_buckets', 'num_latents', and 'latent_dim'
    model = Model(
        num_embeddings=50000,
        embedding_dim=256,
        num_buckets=32,
        num_latents=256,
        latent_dim=32,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        gesture_names=[0, 1, 2, 3, 4, 5],
        learning_rate=1e-4,
        weight_decay=0.01,
    )

    # Train model
    trainer.train(n_epochs=100)

    # Make predictions
    # test_predictions, test_confidences = trainer.predict(_loader)

    # Print some predictions with their confidence
    # for pred, conf in zip(test_predictions[:5], test_confidences[:5]):
    #     print(
    #         f"Predicted gesture: {trainer.gesture_names[pred]} with confidence: {conf:.4f}"
    #     )
