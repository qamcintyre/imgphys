import click
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, ToPILImage, Grayscale
from torch.utils.data import DataLoader


@click.command()
@click.option(
    "--num_images",
    default=100,
    type=int,
    help="Number of images to load from CIFAR-10.",
)
@click.option(
    "--iterations",
    default=10,
    type=int,
    help="Number of Arnold cat map iterations per image.",
)
@click.option(
    "--save_path",
    default="cat_map_sequences.npy",
    type=str,
    help="Path to save the numpy array of sequences.",
)
def main(num_images, iterations, save_path):
    dataset = CIFAR10(root="./data", train=True, download=True, transform=ToTensor())
    loader = DataLoader(dataset, batch_size=num_images, shuffle=False)

    images, _ = next(iter(loader))

    sequences = []
    for image_tensor in images:
        image_pil = ToPILImage()(image_tensor)
        image_gray = Grayscale()(image_pil)
        image_np = np.array(image_gray)
        threshold = 128
        image_np = (image_np > threshold).astype(int)
        sequence = generate_cat_map_sequence(image_np, iterations)
        sequences.append(sequence)

    sequences_array = np.array(sequences)
    np.save(save_path, sequences_array)

    print(f"Sequences saved to {save_path}")


def arnold_cat_map(image_array):
    """
    Apply Arnold's Cat Map to a numpy array representing an image.
    """
    height, width = image_array.shape[:2]
    new_image_array = np.zeros_like(image_array)
    for y in range(height):
        for x in range(width):
            nx = (2 * x + y) % width
            ny = (x + y) % height
            new_image_array[ny, nx] = image_array[y, x]
    return new_image_array


def generate_cat_map_sequence(image, iterations):
    """
    Generate a sequence of images transformed by Arnold's Cat Map.
    """
    sequence = [image]
    for _ in range(iterations - 1):
        image = arnold_cat_map(image)
        sequence.append(image)
    return np.array(sequence)


if __name__ == "__main__":
    main()
