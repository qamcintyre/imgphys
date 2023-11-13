import os
import numpy as np
import matplotlib.pyplot as plt
import click
import imageio 
import io

@click.command()
@click.option('--file_path', type=str, prompt='Path to the video dataset file', 
              help='Path to the .npy file containing the video dataset.')
@click.option('--num_videos', default=1, prompt='Number of videos to display', 
              type=int, help='Number of videos to display from the dataset.')
@click.option('--num_frames', default=10, prompt='Number of frames per video to display', 
              type=int, help='Number of frames to display from each video.')
@click.option('--output_dir', default='output_gifs', prompt='Directory to save the output GIFs', 
              type=str, help='Directory to save the output GIFs.')
def visualize_videos(file_path, num_videos, num_frames, output_dir):
    """
    Saves a specified number of frames from each video in the dataset as GIF files.
    """
    try:
        video_dataset = np.load(file_path)
    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for video_index in range(min(num_videos, len(video_dataset))):
        video = video_dataset[video_index]
        frames = []

        for frame_index in range(min(num_frames, len(video))):
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

        gif_path = os.path.join(output_dir, f'Video_{video_index + 1}.gif')
        imageio.mimsave(gif_path, frames, duration=0.2)  # Adjust duration as needed

if __name__ == '__main__':
    visualize_videos()
