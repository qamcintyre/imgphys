import os
import numpy as np
import cv2
import click

IMAGE_SIZE = 32

@click.command()
@click.option('--num_videos', default=10, type=int, help='Number of videos to generate.')
@click.option('--num_frames', default=100, type=int, help='Number of frames per video.')

def main(num_videos, num_frames):
    """
    Generates and saves frames to the src/data_generation/raw
    """
    save_motion_sequences(num_videos, num_frames)

def generate_motion_sequence(num_frames, size=IMAGE_SIZE):
    """
    Returns a list of frames of the video with random parameters
    """
    # Random initial position, velocity, and size of the circle
    x, y = np.random.randint(0, size, size=2)
    dx, dy = np.random.randint(-3, 4, size=2)  
    radius = np.random.randint(3, 7)  
    
    frames = []
    circle_in_frame = True  
    
    for _ in range(num_frames):
        # Create an empty frame 
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Draw the circle on the frame if it's still within boundaries
        if circle_in_frame:
            cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)  # I made it green

        frames.append(frame)
        
        # Update position 
        x += dx
        y += dy

        # Check if circle has left the frame
        if x + radius < 0 or x - radius >= size or y + radius < 0 or y - radius >= size:
            circle_in_frame = False

    return frames

def save_motion_sequences(num_videos, num_frames_per_video):
    """
    Runs the generation and saves it to the folders
    """
    for i in range(num_videos):
        video_folder = os.path.join("data", "raw", f"video{i+1}")
        os.makedirs(video_folder, exist_ok=True)
        
        frames = generate_motion_sequence(num_frames_per_video)
        
        for j, frame in enumerate(frames):
            frame_path = os.path.join(video_folder, f"frame_{j+1:04}.jpg")
            cv2.imwrite(frame_path, frame)

if __name__ == '__main__':
    main()