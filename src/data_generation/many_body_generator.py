import os
import numpy as np
import click

IMAGE_SIZE = 128  
GRAVITY_CONSTANT = 1.0 
MAX_BODIES = 5  

@click.command()
@click.option('--num_videos', default=100, type=int, help='Number of videos to generate.')
@click.option('--num_frames', default=10, type=int, help='Number of frames per video.')
@click.option('--save_dir', default='data/raw', type=str, help='Directory to save the videos.')

def main(num_videos, num_frames, save_dir):
    save_motion_sequences_as_array(num_videos, num_frames, save_dir)

def generate_motion_sequence(num_frames, size=IMAGE_SIZE):
    num_bodies = np.random.randint(2, MAX_BODIES + 1)  # Random number of bodies
    positions = np.random.uniform(0, size, size=(num_bodies, 2))
    velocities = np.random.uniform(-1, 2, size=(num_bodies, 2))
    edge_lengths = np.random.randint(2, 5, size=num_bodies)  # Random sizes

    frames = []
    for _ in range(num_frames):
        frame = np.zeros((size, size), dtype=np.uint8)
        forces = np.zeros((num_bodies, 2), dtype=np.float64)

        # Gravitational forces and collision detection
        for i in range(num_bodies):
            for j in range(i + 1, num_bodies):
                distance = positions[j] - positions[i]
                distance_magnitude = np.linalg.norm(distance)
                sum_edge_lengths = edge_lengths[i] + edge_lengths[j]
                
                # Collision handling
                if distance_magnitude < sum_edge_lengths:
                    # Simple elastic collision response
                    velocities[i], velocities[j] = velocities[j], velocities[i]

                # Gravitational force calculation
                elif distance_magnitude > 0:
                    force_magnitude = GRAVITY_CONSTANT * edge_lengths[i] * edge_lengths[j] / distance_magnitude**2
                    force_direction = distance / distance_magnitude
                    forces[i] += force_magnitude * force_direction
                    forces[j] -= force_magnitude * force_direction

        # Update velocities and positions
        for i in range(num_bodies):
            velocities[i] += forces[i] / edge_lengths[i]**2  # F = ma, so a = F/m
            positions[i] += velocities[i]

            # Draw each body
            x, y = positions[i]
            edge_length = edge_lengths[i]
            frame[max(0, int(y)):min(size, int(y + edge_length)), max(0, int(x)):min(size, int(x + edge_length))] = 1

        frames.append(frame)

    return frames

def save_motion_sequences_as_array(num_videos, num_frames, save_path):
    video_dataset = []

    for _ in range(num_videos):
        frames = generate_motion_sequence(num_frames)
        video_dataset.append(frames)

    video_dataset = np.array(video_dataset, dtype=np.uint8)
    np.save(save_path, video_dataset)

if __name__ == '__main__':
    main()
