import os
import numpy as np
import click

IMAGE_SIZE = 32

@click.command()
@click.option('--num_videos', default=10, type=int, help='Number of videos to generate.')
@click.option('--num_frames', default=100, type=int, help='Number of frames per video.')
@click.option('--save_dir', default='data/raw', type=str, help='Directory to save the videos.')

def main(num_videos, num_frames, save_dir):
    """
    Generates and saves frames to the specified directory.
    """
    save_motion_sequences(num_videos, num_frames, save_dir)

def generate_motion_sequence(num_frames, size=IMAGE_SIZE):
    """
    Returns a list of frames of the video with random parameters, using arrays.
    """
    # Random initial position, velocity, and size of the square
    x, y = np.random.randint(0, size, size=2)
    dx, dy = np.random.randint(-3, 4, size=2)
    edge_length = np.random.randint(3, 7)

    frames = []
    square_in_frame = True

    for _ in range(num_frames):
        # Create an empty frame
        frame = np.zeros((size, size), dtype=np.uint8)

        # Draw the square on the frame if it's still within boundaries
        if square_in_frame:
            # Make sure the square stays within the frame bounds
            top_left_x = max(0, min(x, size - edge_length))
            top_left_y = max(0, min(y, size - edge_length))
            bottom_right_x = max(0, min(x + edge_length, size))
            bottom_right_y = max(0, min(y + edge_length, size))

            for i in range(top_left_x, bottom_right_x):
                for j in range(top_left_y, bottom_right_y):
                    frame[j, i] = 1  # Set square position to 1

        frames.append(frame)

        # Update position
        x += dx
        y += dy

        # Check if square has left the frame
        if x < 0 or x >= size or y < 0 or y >= size:
            square_in_frame = False

    return frames

def save_motion_sequences(num_videos, num_frames_per_video, save_dir):
    """
    Runs the generation and saves it to the specified folders.
    """
    for i in range(num_videos):
        video_folder = os.path.join(save_dir, f"video_{i+1:04}")
        os.makedirs(video_folder, exist_ok=True)

        frames = generate_motion_sequence(num_frames_per_video)

        for j, frame in enumerate(frames):
            frame_path = os.path.join(video_folder, f"frame_{j+1:04}.txt")
            np.savetxt(frame_path, frame, fmt='%d')

def load_motion_sequences(frame_path):
    """
    Loads saved generations.
    """
    video_dataset = []
    for video_folder in sorted(os.listdir(frame_path)):
        video = []
        full_folder_path = os.path.join(frame_path, video_folder)
        for frame in sorted(os.listdir(full_folder_path)):
            frame_data = np.loadtxt(os.path.join(full_folder_path, frame), dtype=np.uint8)
            video.append(frame_data)
        video_dataset.append(np.stack(video))
    return np.stack(video_dataset)

if __name__ == '__main__':
    main()
