import os
import numpy as np
import click

IMAGE_SIZE = 128  
GRAVITY_CONSTANT = 1.0 
MAX_BODIES = 8  
<<<<<<< HEAD
DAMPING_FACTOR = 0.99  # Damping factor to stabilize motion
MAX_VELOCITY_CHANGE = 0.5  # Maximum change in velocity per frame for smoother motion
=======
>>>>>>> 87d2975b7883ff3ae4a9fbf531b4326c76e27a03

@click.command()
@click.option('--num_videos', default=100, type=int, help='Number of videos to generate.')
@click.option('--num_frames', default=10, type=int, help='Number of frames per video.')
@click.option('--save_dir', default='data/raw', type=str, help='Directory to save the videos.')

def main(num_videos, num_frames, save_dir):
    save_motion_sequences_as_array(num_videos, num_frames, save_dir)

def generate_motion_sequence(num_frames, size=IMAGE_SIZE):
<<<<<<< HEAD
    num_bodies = np.random.randint(2, MAX_BODIES + 1)
    positions = np.random.uniform(0, size, size=(num_bodies, 2))
    velocities = np.random.uniform(-1, 1, size=(num_bodies, 2))
    edge_lengths = np.random.randint(2, 9, size=num_bodies)
=======
    num_bodies = np.random.randint(2, MAX_BODIES + 1)  # Random number of bodies
    positions = np.random.uniform(0, size, size=(num_bodies, 2))
    velocities = np.random.uniform(-1, 2, size=(num_bodies, 2))
    edge_lengths = np.random.randint(2, 9, size=num_bodies)  # Random sizes
>>>>>>> 87d2975b7883ff3ae4a9fbf531b4326c76e27a03

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
<<<<<<< HEAD
                    # Elastic collision response
=======
                    # Simple elastic collision response
>>>>>>> 87d2975b7883ff3ae4a9fbf531b4326c76e27a03
                    velocities[i], velocities[j] = velocities[j], velocities[i]

                # Gravitational force calculation
                elif distance_magnitude > 0:
                    force_magnitude = GRAVITY_CONSTANT * edge_lengths[i] * edge_lengths[j] / distance_magnitude**2
                    force_direction = distance / distance_magnitude
                    forces[i] += force_magnitude * force_direction
                    forces[j] -= force_magnitude * force_direction

<<<<<<< HEAD
                # Update velocities and positions
        for i in range(num_bodies):
            # Limit the change in velocity for smoother motion
            velocity_change = forces[i] / edge_lengths[i]**2
            velocity_change = np.clip(velocity_change, -MAX_VELOCITY_CHANGE, MAX_VELOCITY_CHANGE)
            velocities[i] += velocity_change
            velocities[i] *= DAMPING_FACTOR  # Apply damping
            positions[i] += velocities[i]

            # Clamp positions to ensure they stay within the frame
            positions[i] = np.clip(positions[i], 0, size - edge_lengths[i])

            # Draw each body, ensuring only the visible part is rendered
            x, y = positions[i]
            edge_length = edge_lengths[i]
            x0, x1 = max(0, int(x)), min(size, int(x + edge_length))
            y0, y1 = max(0, int(y)), min(size, int(y + edge_length))
            frame[y0:y1, x0:x1] = 1

=======
        # Update velocities and positions
        for i in range(num_bodies):
            velocities[i] += forces[i] / edge_lengths[i]**2  # F = ma, so a = F/m
            positions[i] += velocities[i]

            # Draw each body
            x, y = positions[i]
            edge_length = edge_lengths[i]
            frame[max(0, int(y)):min(size, int(y + edge_length)), max(0, int(x)):min(size, int(x + edge_length))] = 1
>>>>>>> 87d2975b7883ff3ae4a9fbf531b4326c76e27a03

        frames.append(frame)

    return frames

def save_motion_sequences_as_array(num_videos, num_frames, save_path):
    video_dataset = []

    for _ in range(num_videos):
        frames = generate_motion_sequence(num_frames)
        video_dataset.append(frames)

    video_dataset = np.array(video_dataset, dtype=np.uint8)
<<<<<<< HEAD
    if not os.path.exists(save_path):
        os.makedirs(save_path)
=======
>>>>>>> 87d2975b7883ff3ae4a9fbf531b4326c76e27a03
    np.save(save_path, video_dataset)

if __name__ == '__main__':
    main()
<<<<<<< HEAD





# import os
# import numpy as np
# import click

# IMAGE_SIZE = 128  
# GRAVITY_CONSTANT = 1.0 
# MAX_BODIES = 8  
# DAMPING_FACTOR = 0.99  # Damping factor to stabilize motion

# @click.command()
# @click.option('--num_videos', default=100, type=int, help='Number of videos to generate.')
# @click.option('--num_frames', default=10, type=int, help='Number of frames per video.')
# @click.option('--save_dir', default='data/raw', type=str, help='Directory to save the videos.')

# def main(num_videos, num_frames, save_dir):
#     save_motion_sequences_as_array(num_videos, num_frames, save_dir)

# def generate_motion_sequence(num_frames, size=IMAGE_SIZE):
#     num_bodies = np.random.randint(2, MAX_BODIES + 1)
#     positions = np.random.uniform(0, size, size=(num_bodies, 2))
#     velocities = np.random.uniform(-1, 1, size=(num_bodies, 2))
#     edge_lengths = np.random.randint(2, 9, size=num_bodies)

#     frames = []
#     for _ in range(num_frames):
#         frame = np.zeros((size, size), dtype=np.uint8)
#         forces = np.zeros((num_bodies, 2), dtype=np.float64)

#         # Gravitational forces and collision detection
#         for i in range(num_bodies):
#             for j in range(i + 1, num_bodies):
#                 distance = positions[j] - positions[i]
#                 distance_magnitude = np.linalg.norm(distance)
#                 sum_edge_lengths = edge_lengths[i] + edge_lengths[j]
                
#                 # Collision handling
#                 if distance_magnitude < sum_edge_lengths:
#                     # Elastic collision response
#                     velocities[i], velocities[j] = velocities[j], velocities[i]
#                     # Position adjustment to prevent overlap
#                     overlap = sum_edge_lengths - distance_magnitude
#                     positions[i] -= overlap * (distance / distance_magnitude)
#                     positions[j] += overlap * (distance / distance_magnitude)

#                 # Gravitational force calculation
#                 elif distance_magnitude > 0:
#                     force_magnitude = GRAVITY_CONSTANT * edge_lengths[i] * edge_lengths[j] / distance_magnitude**2
#                     force_direction = distance / distance_magnitude
#                     forces[i] += force_magnitude * force_direction
#                     forces[j] -= force_magnitude * force_direction

#         # Update velocities and positions
#         for i in range(num_bodies):
#             velocities[i] += forces[i] / edge_lengths[i]**2
#             velocities[i] *= DAMPING_FACTOR  # Apply damping
#             positions[i] += velocities[i]

#             # Draw each body
#             x, y = positions[i]
#             edge_length = edge_lengths[i]
#             frame[max(0, int(y)):min(size, int(y + edge_length)), max(0, int(x)):min(size, int(x + edge_length))] = 1

#         frames.append(frame)

#     return frames

# def save_motion_sequences_as_array(num_videos, num_frames, save_path):
#     video_dataset = []

#     for _ in range(num_videos):
#         frames = generate_motion_sequence(num_frames)
#         video_dataset.append(frames)

#     video_dataset = np.array(video_dataset, dtype=np.uint8)
#     np.save(save_path, video_dataset)

# if __name__ == '__main__':
#     main()
=======
>>>>>>> 87d2975b7883ff3ae4a9fbf531b4326c76e27a03
