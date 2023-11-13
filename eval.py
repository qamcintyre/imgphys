from src.globals import default_config
from copy import deepcopy
import math
import numpy as np
import lightning.pytorch as pl
from src.system import ConvLSTMTrainingSystem
import torch
from src.data import DataModule
import matplotlib.pyplot as plt
import os
import io
import imageio
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
    
def make_gif(video, gif_path):
        frames = []

        for frame_index in range(video.shape[0]):
            fig, ax = plt.subplots()
            ax.imshow(video[frame_index], cmap='gray')
            ax.axis('off')
            # Save frame to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            frame = imageio.imread(buf)
            frames.append(frame)
            buf.close()

        imageio.mimsave(gif_path, frames, duration=0.2)  # Adjust duration as needed
        
def visualize_outputs(model, data_module, wandb_config):
    val_loader = data_module.train_dataloader()
    for single_batch in val_loader:
        inputs, _ = single_batch
        for i in range(inputs.shape[0]):
            single_input = inputs[i].unsqueeze(0) 
            start_idx = inputs.shape[2]
            model.eval()
            with torch.no_grad():
                output = model.test_step(single_input, num_generations = wandb_config["num_generations"])

            gif_path = os.path.join(wandb_config['save_path'], f'Video_{i }.gif')
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
            make_gif(output.squeeze().cpu().numpy(), gif_path)

if __name__ == "__main__":
    model = ConvLSTMTrainingSystem(default_config, wandb_logger=None) 
    model.load_state_dict(torch.load(default_config["model_path"])["state_dict"])

    data_module = DataModule(
        wandb_config=default_config, 
        run_checkpoint_dir=default_config["model_path"]
    )
    data_module.setup('test')
    # evaluate_model(model, data_module, default_config)
    visualize_outputs(model, data_module, default_config)