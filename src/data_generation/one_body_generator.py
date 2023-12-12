import os
import numpy as np
import click

IMAGE_SIZE = 32


@click.command()
@click.option(
    "--num_videos", default=100, type=int, help="Number of videos to generate."
)
@click.option("--num_frames", default=10, type=int, help="Number of frames per video.")
@click.option(
    "--save_dir", default="data/raw", type=str, help="Directory to save the videos."
)
def main(num_videos, num_frames, save_dir):
    """
    Generates and saves frames to the specified directory.
    """
    save_motion_sequences_as_array(num_videos, num_frames, save_dir)


def generate_motion_sequence(num_frames, size=IMAGE_SIZE):
    """
    Returns a list of frames of the video with random parameters, using arrays.
    """
    edge_length = np.random.randint(3, 7)
    safe_max = size - edge_length - 2
    x, y = np.random.randint(0, safe_max, size=2)
    dx, dy = np.random.randint(-3, 4, size=2)
    frames = []
    for _ in range(num_frames):
        frame = np.zeros((size, size), dtype=np.uint8)
        if x + edge_length > 0 and y + edge_length > 0 and x < size and y < size:
            adjusted_x = max(0, x)
            adjusted_y = max(0, y)
            adjusted_edge_x = min(x + edge_length, size)
            adjusted_edge_y = min(y + edge_length, size)
            frame[adjusted_y:adjusted_edge_y, adjusted_x:adjusted_edge_x] = 1
        frames.append(frame)
        x += dx
        y += dy
    return frames


def save_motion_sequences_as_array(num_videos, num_frames_per_video, save_path):
    """
    Saves the generated videos as a single NumPy array.
    """
    video_dataset = []

    for _ in range(num_videos):
        frames = generate_motion_sequence(num_frames_per_video)
        video_dataset.append(frames)

    video_dataset = np.array(video_dataset, dtype=np.uint8)
    np.save(save_path, video_dataset)


if __name__ == "__main__":
    main()
