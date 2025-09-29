import numpy as np
import scipy.constants as const
from main import System

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D # This is necessary for the '3d' projection

num_of_particles = 9

p = np.array([[-7.96712825e-03, -2.90611166e-03, 2.10213120e-04],
 [-2.82597500e-01, 1.97456095e-01, 4.17742433e-02],
 [-7.23209543e-01, -7.94829045e-02, 4.04286220e-02],
 [-1.73818374e-01, 9.66324671e-01, 1.55297876e-04],
 [-3.01325412e-01, -1.45402922e+00, -2.30054066e-02],
 [ 3.48520330e+00, 3.55213702e+00, -9.27104467e-02],
 [ 8.98810505e+00, -3.71906474e+00, -2.93193870e-01],
 [ 1.22630250e+01, 1.52973880e+01, -1.02054995e-01],
 [ 2.98350154e+01, -1.79381284e+00, -6.50640206e-01]]) * const.au

v =np.array([[4.87524241e-06, -7.05716139e-06, -4.57929038e-08],
 [-2.23216589e-02, -2.15720711e-02, 2.85519283e-04],
 [ 2.03406835e-03, -2.02082863e-02, -3.94564043e-04],
 [-1.72300122e-02, -2.96772137e-03, 6.38154172e-07],
 [ 1.42483227e-02, -1.57923621e-03, -3.82372338e-04],
 [-5.47097051e-03, 5.64248731e-03, 9.89618477e-05],
 [ 1.82201399e-03, 5.14347040e-03, -1.61723649e-04],
 [-3.09761521e-03, 2.27678190e-03, 4.86042739e-05],
 [ 1.67653809e-04, 3.15209870e-03, -6.87750693e-05]]) * const.au / const.day

m = np.array([[1.00000000e+00], 
              [1.66012083e-07], 
              [2.44783829e-06], 
              [3.00348962e-06],
              [3.22715608e-07], 
              [9.54791910e-04], 
              [2.85885670e-04], 
              [4.36624961e-05],
              [5.15138377e-05]]) * 1.9885e30

dt = 86400
T= 3.154e7 * 165

sys = System(m, v, p, const.G)
history = sys.simulate(m, p, v, t=0, dt=dt, T=T)
history_p = history[:, :, 0, :]

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
ani = FuncAnimation(fig, update, frames=num_frames, interval = 1, blit=False)

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