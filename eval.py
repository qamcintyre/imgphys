from src.globals import default_config
from copy import deepcopy
import math
import numpy as np
import lightning.pytorch as pl
from src.system import TrainingSystem
import torch
from src.data import DataModule
import matplotlib.pyplot as plt
import os
import io
import imageio
from tqdm import tqdm


def evaluate_model(model, data_module, wandb_config):
    val_loader = data_module.train_dataloader()
    model.eval()

    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model.model(inputs)

            loss, accuracy = model._calculate_loss(outputs, targets)
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    print(f"Average Validation Loss: {avg_loss}")
    print(f"Average Validation Accuracy: {avg_accuracy}")


# def make_gif(video, gif_path):
#         frames = []

#         for frame_index in range(video.shape[0]):
#             fig, ax = plt.subplots()
#             ax.imshow(video[frame_index], cmap='gray')
#             ax.axis('off')
#             # Save frame to buffer
#             buf = io.BytesIO()
#             plt.savefig(buf, format='png')
#             plt.close(fig)
#             buf.seek(0)
#             frame = imageio.imread(buf)
#             frames.append(frame)
#             buf.close()

#         imageio.mimsave(gif_path, frames, duration=0.2)  # Adjust duration as needed


def make_gif(video, grid_path):
    grid_size = 4
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < video.shape[0]:
                axs[i, j].imshow(video[idx], cmap="gray")
            axs[i, j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(grid_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def visualize_outputs(model, data_module, wandb_config):
    train_loader = data_module.train_dataloader()
    n_batches = 0
    for single_batch in train_loader:
        n_batches += 1
        inputs, _ = single_batch
        for i in tqdm(range(inputs.shape[0])):
            single_input = inputs[i].unsqueeze(0)
            start_idx = inputs.shape[2]
            model.eval()
            with torch.no_grad():
                output = model.test_step(
                    single_input, num_generations=wandb_config["num_generations"]
                )
                true_output = torch.cat(
                    (single_input.squeeze(), output.squeeze()), dim=0
                )

            gif_path = os.path.join(wandb_config["save_path"], f"Video_{i }.png")
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            make_gif(true_output.squeeze(), gif_path)
        break


if __name__ == "__main__":
    model = TrainingSystem(default_config, wandb_logger=None)
    model.load_state_dict(torch.load(default_config["model_path"])["state_dict"])

    data_module = DataModule(
        wandb_config=default_config, run_checkpoint_dir=default_config["model_path"]
    )
    data_module.setup("test")
    # evaluate_model(model, data_module, default_config)
    visualize_outputs(model, data_module, default_config)
