import datetime
import gzip
import os
import pickle
import h5py
import joblib
import numpy as np
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

import deepspeed
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def setup_distributed():
    # deepspeed.init_distributed()
    timeout = datetime.timedelta(hours=2)
    # Initialize the process group manually
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timeout
    )
    rank = dist.get_rank()  # Unique ID for each GPU (0 to num_gpus-1)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Device index
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, rank
torch.autograd.set_detect_anomaly(True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        gesture_names,
        key_idx_mapping,
        learning_rate=1e-4,
        weight_decay=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ds_config_path="ds_config.json",
        
        # device="cpu",
    ):

        # set member variables
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gesture_names = gesture_names
        self.device = device
        self.ds_config_path = ds_config_path
        self.idx_key_mapping = {value: key for key, value in key_idx_mapping.items()}

        # initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )

        self.model, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            model_parameters=self.model.parameters(),
            config=self.ds_config_path,
            dist_init_required=True
        )
        print(f"Model is on: {next(self.model.parameters()).device}")

        self.latent_idx, self.latent_timestamps = create_linspace_latent_tokens(
            0, 1.0, 0.125, 32
        )

    def train_epoch(self, epoch):
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0
        all_preds = []
        all_labels = []
        loss_arr = []
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f"Training (Epoch {epoch})", disable=(dist.get_rank() != 0))
        
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

            sessions = sessions.to(self.device, non_blocking=True)
            subjects = subjects.to(self.device, non_blocking=True)
            channels = channels.to(self.device, non_blocking=True)
            prominences = prominences.to(self.device, non_blocking=True)
            durations = durations.to(self.device, non_blocking=True)
            timestamps = timestamps.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)
            labels = labels.to(torch.long)
            labels = labels.to(self.device, non_blocking=True)

            features = torch.stack(
                (sessions, subjects, channels, prominences, durations), dim=1
            ).to(self.device)
            # print("FEATURES SHAPE BEFORE TRANSPOSE", features.shape)
            features = torch.transpose(features, 1, 2)
            # print("FEATURES SHAPE AFTER TRANSPOSE", features.shape)
            # zero gradients
            self.optimizer.zero_grad()

            # print(f"Feature tensor is on: {features.device}")
            # forward pass!
            with torch.cuda.amp.autocast(enabled=False):
                # print("inside with torch.cuda.amp statement")
                predictions, loss = self.model(
                    data=features,
                    sequence_lengths=lengths,
                    time_stamps=timestamps,
                    latent_timestamps=self.latent_timestamps,
                    latent_idx=self.latent_idx,
                    labels=labels,
                )
            loss = loss.to(self.device)
            # pred_label = torch.argmax(predictions, dim=1)
            # correct = (pred_label == labels).sum().item()
            # total_correct += correct
            # total_samples += labels.size(0)
            # train_accuracy = correct / labels.size(0)

            self.model.backward(loss)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.model.step()

            running_loss += loss.item()
            loss_arr.append(loss.item())
            self.plot_train_loss(loss_arr)
            pred_labels = torch.argmax(predictions, dim=1)
            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        dist.barrier()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

        with open(MODEL_SAVE_PATH + "model_metrics", "a") as file:
            file.write("training metrics:")
            metric_line = "loss: " + str(epoch_loss) + " accuracy: " + str(epoch_acc) + " f1 " + str(epoch_f1) + "\n"
            file.write(metric_line)

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

        with open(MODEL_SAVE_PATH + "model_metrics", "a") as file:
            file.write("validation metrics:")
            metric_line = "loss: " + str(val_loss) + " accuracy: " + str(val_acc) + " f1 " + str(val_f1) + "\n"
            file.write(metric_line)

        return {"loss": val_loss, "accuracy": val_acc, "f1": val_f1}

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[self.idx_key_mapping[idx] for idx in self.gesture_names],
            yticklabels=[self.idx_key_mapping[idx] for idx in self.gesture_names],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(MODEL_SAVE_PATH + "confusion_mat.png")
        plt.close()

    def plot_train_loss(self, loss_arr):
        plt.figure(figsize=(10, 8))
        plt.plot(loss_arr)
        plt.title("Training Loss")
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
        plt.savefig(MODEL_SAVE_PATH + "train_loss.png")
        plt.close()


    def plot_val_loss(self, loss_arr):
        plt.figure(figsize=(10, 8))
        plt.plot(loss_arr)
        plt.title("Validation Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(MODEL_SAVE_PATH + "val_loss.png")
        plt.close()

    def train(self, n_epochs):

        loss_arr = []
        for epoch in range(n_epochs):
            print("epoch number:", epoch)

            train_metrics = self.train_epoch(epoch)

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

            loss_arr.append(val_metrics["loss"])
            self.plot_val_loss(loss_arr)

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
    device, rank = setup_distributed()
    print("LOCAL RANK", rank)
    embedding_dim = 256 #was 256
    session_emb_dim = 8
    subject_emb_dim = 8
    #need session_idx, stage_idx (key idx), subject_idx, and data
    DATA_STORE = "data_3-6-2025/"
    # TRAIN_DATA_PATH = "torch_data/train_data.pt"
    # VAL_DATA_PATH = "torch_data/val_data.pt"
    # TRAIN_SPIKE_DATASET_PATH = "torch_data/train_spike_token_data.pt"
    # VAL_SPIKE_DATASET_PATH = "torch_data/val_spike_token_data.pt"
    MODEL_SAVE_PATH = "3-6-2025/"
    # DATA_BATCH_SIZE = 50
    # NUM_TRAIN_FILES = 200
    # NUM_VAL_FILES = 50
    # DATA_FRESH = False
    stage_idx = {}
    session_idx = {}
    subject_idx = {}
    train_data = None
    val_data = None
    device = torch.cuda.current_device()
    print(f"PyTorch sees {torch.cuda.device_count()} GPUs")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    if rank == 0:
        if os.path.exists(DATA_STORE + "key_idx.pickle.gz"):
            with gzip.open(DATA_STORE + "key_idx.pickle.gz", "rb") as f:
                stage_idx = pickle.load(f)
                print("SUCCESS", rank)
        if os.path.exists(DATA_STORE + "session_idx.pickle.gz"):
            with gzip.open(DATA_STORE + "session_idx.pickle.gz", "rb") as f:
                session_idx = pickle.load(f)
                print("SUCCESS", rank)
        if os.path.exists(DATA_STORE + "subject_idx.pickle.gz"):
            with gzip.open(DATA_STORE + "subject_idx.pickle.gz", "rb") as f:
                subject_idx = pickle.load(f)
                print("SUCCESS", rank)
        
        
    obj_list = [stage_idx, session_idx, subject_idx]
    dist.broadcast_object_list(obj_list, src=0)
    stage_idx, session_idx, subject_idx = obj_list 
        
    print("BROADCASTED EVERTHING", rank)
    dist.barrier()
    labels = set(stage_idx.values())

    # if local_rank == 0:
    train_dataset_input = []

    with h5py.File(DATA_STORE + "train_input_tensor.h5", "r") as file:
        num_sublists = file.attrs["num_sublists"]
        for i in range(num_sublists):
            train_dataset_input.append(file[f"sublist_{i}"][:])  # Load entire 2D array
    # # torch.save(train_data, TRAIN_DATA_PATH) #the tensor of all train data

    val_dataset_input = []

    with h5py.File(DATA_STORE + "val_input_tensor.h5", "r") as file:
        num_sublists = file.attrs["num_sublists"]
        for i in range(num_sublists):
            val_dataset_input.append(file[f"sublist_{i}"][:])  # Load entire 2D array
        
    print("create SpikeDataset train")
    train_spike_token_data = SpikeDataset(train_dataset_input)
    print("Length of train_spike_token_data", len(train_spike_token_data))
    print("create SpikeDataset val")
    val_spike_token_data = SpikeDataset(val_dataset_input)
    print("Length of val_spike_token_data", len(val_spike_token_data))

    # ✅ Create `DistributedSampler` on **every rank** (not just rank 0)
    print("create distributed sampler")
    print(f"Rank {dist.get_rank()} - Total dataset size: {len(train_spike_token_data)}")
    train_sampler = DistributedSampler(train_spike_token_data, num_replicas=dist.get_world_size(), rank=rank, drop_last=True)
    val_sampler = DistributedSampler(val_spike_token_data, num_replicas=dist.get_world_size(), rank=rank, drop_last=True)
    print(f"Rank {dist.get_rank()} - Number of samples assigned: {len(train_sampler)}")

    # ✅ Each rank gets its subset of data
    print("create dataloader train")
    train_loader = DataLoader(
        dataset=train_spike_token_data,
        batch_size=32,
        sampler=train_sampler,
        num_workers=16, #was 4
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        collate_fn=SpikeDataset.collate_fn
    )
    print("create dataloader val")
    val_loader = DataLoader(
        dataset=val_spike_token_data,
        batch_size=32,
        sampler=val_sampler,
        num_workers=16, #was 4
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        collate_fn=SpikeDataset.collate_fn
    )

    # ✅ Barrier before training starts
    torch.distributed.barrier()

    

    # # print("amount of data in training", len(train_spike_token_data))
    # train_sampler = DistributedSampler(train_spike_token_data)
    # print("creating train dataloader", local_rank)
    # train_loader = DataLoader(
    #     train_spike_token_data,
    #     batch_size=8,  
    #     sampler=train_sampler,
    #     shuffle=True,
    #     collate_fn=SpikeDataset.collate_fn,
    # )
    # print("creating val dataloader", local_rank)
    # val_loader = DataLoader(
    #     val_spike_token_data,
    #     batch_size=8,
    #     shuffle=False,
    #     collate_fn=SpikeDataset.collate_fn,
    # )
    # need parameters 'num_embeddings', 'embedding_dim', 'num_buckets', 'num_latents', and 'latent_dim'
    print("creating model", rank)
    print("NUM CLASSES (len(list(stage_idx.values())))", len(list(stage_idx.values())))
    model = Model(
        embedding_dim=embedding_dim,
        session_emb_dim=8,
        subject_emb_dim=8,
        num_latents=256, #was 256
        latent_dim=256, #was 256
        num_classes=len(list(stage_idx.values())),
        emb_directory=DATA_STORE,
        device=device,
    ).to(device)
    # model.double()
    # Initialize trainer
    print("init trainer", rank)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        gesture_names=list(stage_idx.values()),
        learning_rate=1e-4,
        weight_decay=1e-3,
        key_idx_mapping=stage_idx
    )

    # Train model
    print("training", rank)
    trainer.train(n_epochs=20)

    # Make predictions
    # test_predictions, test_confidences = trainer.predict(_loader)

    # Print some predictions with their confidence
    # for pred, conf in zip(test_predictions[:5], test_confidences[:5]):
    #     print(
    #         f"Predicted gesture: {trainer.gesture_names[pred]} with confidence: {conf:.4f}"
    #     )
