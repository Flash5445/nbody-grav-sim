from main import System
import numpy as np
import scipy.constants as const

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D # This is necessary for the '3d' projection

num_of_particles = 4
m = np.random.rand(num_of_particles, 1) * 1e12
v = np.zeros((num_of_particles, 3))
p = np.random.rand(num_of_particles, 3) * 1e2
dt = 0.1
T = 16

sys = System(m, v, p, const.G)
history = sys.simulate(m, p, v, t=0, dt=dt, T=T)
history_p = history[:, :, 0, :]
history_v = history[:, :, 1, :]
print(history_p.shape, history_v.shape)

# --- 1. Load or Generate Your Data ---
# Replace this section with your actual data.
# The data should be a NumPy array with the shape (frames, points, dimensions)
# For this example, that is (100, 10, 3).

print("Generating sample data...")
num_frames = history_p.shape[0]
num_points = history_p.shape[1]
data = history_p
print(f"Data shape: {data.shape}")

# --- 2. Set up the Figure and 3D Axes ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# --- 3. Initialize the Scatter Plot ---
# We'll start by plotting the first frame of data (data[0]).
# The 'scatter' object will be updated in each frame of the animation.
scatter = ax.scatter(data[0, :, 0], data[0, :, 1], data[0, :, 2], marker='o', s=50)

# --- 4. Set Axis Limits and Labels ---
# To prevent the plot from rescaling on every frame, we calculate the
# limits from the entire dataset.
ax.set_xlim(data[:, :, 0].min(), data[:, :, 0].max())
ax.set_ylim(data[:, :, 1].min(), data[:, :, 1].max())
ax.set_zlim(data[:, :, 2].min(), data[:, :, 2].max())

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Point Animation')

# --- 5. Define the Animation Update Function ---
# This function will be called for each frame of the animation.
def update(frame):
    """
    Updates the positions of the scatter points for the current frame.
    
    Args:
        frame (int): The current frame number passed by FuncAnimation.
    """
    # Get the positions for the current frame
    current_positions = data[frame]
    
    # Update the scatter plot's data. For 3D plots, we update the '_offsets3d'
    # property, which stores the (x, y, z) coordinates.
    scatter._offsets3d = (current_positions[:, 0], current_positions[:, 1], current_positions[:, 2])
    
    # Update the title to show the current time step
    time = frame * dt # Since each frame is a 0.1s step
    ax.set_title(f'3D Point Animation (Time: {time:.1f}s)')
    
    return scatter,

# --- 6. Create and Run the Animation ---
# FuncAnimation combines the figure, the update function, and the frames
# to create the animation.
# - fig: The figure object to draw on.
# - update: The function to call for each frame.
# - frames: The number of frames (in this case, 100).
# - interval: Delay between frames in milliseconds.
ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

# Show the plot
plt.show()

# --- Optional: Save the Animation ---
# To save the animation, you may need to install 'ffmpeg' or 'imagemagick'.
# You can install ffmpeg with: conda install -c conda-forge ffmpeg
# Then, uncomment one of the lines below.

# print("Saving animation as MP4...")
#ani.save('3d_point_animation.gif', writer='ffmpeg', fps=20)
#print("Done.")

# print("Saving animation as GIF...")
# ani.save('3d_point_animation.gif', writer='imagemagick', fps=20)
# print("Done.")