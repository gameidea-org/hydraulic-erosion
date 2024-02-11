import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Set the size of the 2D array (adjust these values as needed)
width = 128
height = 128

def create_heightmap(width: int, height: int, scale: float = 32.0, seed: int = 42):
    # Create a 2D array of floats
    heightmap = np.zeros((width, height), dtype=float)

    # Create a Perlin noise object
    noise_generator = OpenSimplex(seed=seed)

    # Generate Perlin noise and assign it to the heightmap
    for i in range(height):
        for j in range(width):
            heightmap[i][j] = noise_generator.noise2(i / scale, j / scale)
            heightmap[i][j] += noise_generator.noise2(i / scale / 2, j / scale / 2) * 0.5
            heightmap[i][j] += noise_generator.noise2(i / (scale / 4), j / (scale / 4)) * 0.25
            heightmap[i][j] += noise_generator.noise2(i / (scale / 8), j / (scale / 8)) * 0.125
            heightmap[i][j] += noise_generator.noise2(i / (scale / 16), j / (scale / 16)) * 0.0625
            heightmap[i][j] += noise_generator.noise2(i / (scale / 32), j / (scale / 32)) * 0.03125

    # Normalize the heightmap to the range [0, 1]
    heightmap -= heightmap.min()
    heightmap /= heightmap.max()

    return heightmap

heightmap = create_heightmap(width, height, seed=56456)

def gaussian_blur(heightmap, radius):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
    kernel /= np.sum(kernel)

    for _ in range(radius):
        heightmap = np.pad(heightmap, 1, mode='edge')
        heightmap = np.array([[np.sum(heightmap[i:i+3, j:j+3] * kernel) for j in range(width)] for i in range(height)])

    return heightmap


def get_normal(heightmap, position):
    x, y = position

    if x == 0 or x >= width-1 or y == 0 or y >= height-1:
        return np.array([0, 0, 1], dtype=float)

    R = heightmap[int(x+1), int(y)]
    L = heightmap[int(x-1), int(y)]
    T = heightmap[int(x), int(y+1)]
    B = heightmap[int(x), int(y-1)]

    dx = (R - L) * 0.5
    dy = (B - T) * 0.5
    dz = -1.0

    normal = np.array([dx, dy, dz], dtype=float)
    normal /= np.linalg.norm(normal)

    return normal

def interpolated_normal(heightmap, position):
    x, y = position

    if x == 0 or x >= width-1 or y == 0 or y >= height-1:
        return np.array([0, 0, 1], dtype=float)

    R = get_normal(heightmap, [x+1, y])
    L = get_normal(heightmap, [x-1, y])
    T = get_normal(heightmap, [x, y+1])
    B = get_normal(heightmap, [x, y-1])

    # more distant neighbors as well
    TR = get_normal(heightmap, [x+1, y+1])
    TL = get_normal(heightmap, [x-1, y+1])
    BR = get_normal(heightmap, [x+1, y-1])
    BL = get_normal(heightmap, [x-1, y-1])


    return (R + L + T + B + TR + TL + BR + BL) / 8


class Drop:
    velocity = np.array([0, 0], dtype=float)

    old_position = np.array([0, 0], dtype=int)
    current_position = np.array([0, 0], dtype=int)

    sediment = 0.0

    deposition_rate: float = 0.021  # const
    erosion_rate: float = 0.051  # const
    iteration_scale: float = 0.01  # const
    friction: float = 0.01  # const
    speed: float = 0.99610  # const


    def __init__(self, position, velocity=np.array([0, 0], dtype=float)):
        self.current_position = position
        self.velocity = velocity

    def erode(self, heightmap, max_iterations=6):
        for i in range(max_iterations):
            normal = interpolated_normal(heightmap, self.current_position)
            if normal[2] >= 0.9:
                break  # If the terrain is flat, stop simulating
            deposit = self.sediment * self.deposition_rate * normal[2]
            erosion = self.erosion_rate * (1 - normal[2]) * min(1, i * self.iteration_scale)

            heightmap[int(self.old_position[0]), int(self.old_position[1])] += (deposit - erosion)
            self.sediment += (erosion - deposit)

            self.velocity[0] = self.friction * self.velocity[0] + normal[0] * self.speed
            self.velocity[1] = self.friction * self.velocity[1] + normal[1] * self.speed

            self.old_position = self.current_position
            self.current_position = self.current_position + self.velocity

# Create a figure and axis for the animation
fig, ax = plt.subplots()
im = ax.imshow(heightmap, cmap='inferno', interpolation='none')

drops = 1000
# Animation update function
def update(frame):
    for _ in range(drops):
        drop = Drop(np.array([np.random.randint(0, width), np.random.randint(0, height)], dtype=int))
        drop.erode(heightmap)
    #gaussian_blur(heightmap, 2)
    
    # Update the image data for the animation
    im.set_array(heightmap)

# Keep playing the animation indefinitely
ani = FuncAnimation(fig, update, interval=10)
ax.set_title('Erosion Simulation')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()