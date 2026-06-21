import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
from kinematics import joint_positions, forward_kinematics
from planning import choose_trajectory
from config import D2, D3, D4, H2, H3

REACH = D2 + D3 + D4 #arm's max reach
STEPS = 10 #frames per move
THRESHOLD = 0.1 #max joint jump before cartesian planning is rejected
Z_FIXED = H2 + H3 #lift at h1 = 0 for 2D demo
GAMMA_FIXED = 0.0 #tool orientation fixed

#scene
fig, ax = plt.subplots(figsize=(7, 7))
fig.subplots_adjust(bottom=0.30, top=0.95)
ax.set_aspect("equal")
ax.set_xlim(-REACH * 1.1, REACH * 1.1) #fixed limits
ax.set_ylim(-REACH * 1.1, REACH * 1.1)
ax.grid(True, alpha=0.3)
ax.plot(0, 0, "ks", markersize=8)

bones, = ax.plot([], [], "-o", linewidth=3, markersize=8)
trail, = ax.plot([], [], ".", markersize=2, alpha=0.6)
target_marker, = ax.plot([], [], "rx", markersize=12, markeredgewidth=2)

#sliders
ax_t1 = fig.add_axes([0.25, 0.18, 0.55, 0.03])
ax_t2 = fig.add_axes([0.25, 0.12, 0.55, 0.03])
ax_t3 = fig.add_axes([0.25, 0.06, 0.55, 0.03])

s1 = Slider(ax_t1, "theta1", -180, 180, valinit=0)
s2 = Slider(ax_t2, "theta2", -180, 180, valinit=180)
s3 = Slider(ax_t3, "theta3", -180, 180, valinit=-180)

#state
current_config = (*np.radians([0, 180, -180]), 0.0)
anim = None

def draw_config(config):
    pts = joint_positions(config[:3], h1=0)
    bones.set_data(pts[:, 0], pts[:, 1])

draw_config(current_config)

def on_click(event):
    global current_config, anim
    if event.inaxes != ax:
        return
    
    target_marker.set_data([event.xdata], [event.ydata])
    gamma = np.arctan2(event.ydata, event.xdata)
    target_pose = (event.xdata, event.ydata, Z_FIXED, gamma)
    trajectory, mode = choose_trajectory(current_config, target_pose, STEPS, THRESHOLD)
    ax.set_title(f"mode: {mode}")

    if trajectory is None:
        fig.canvas.draw_idle()
        return 
    
    trail.set_data([], []) #path for movement

    def frame(i):
        config = trajectory[i]
        draw_config(config)
        tip = joint_positions(config[:3], h1=0)[-1] #tool point P3
        xs, ys = trail.get_data()
        trail.set_data([*xs, tip[0]], [*ys, tip[1]]) #grow the trail
        return bones, trail
    
    anim = FuncAnimation(fig, frame, frames=len(trajectory),
                         interval=20, blit=False, repeat=False)
    
    current_config = tuple(trajectory[-1])  #remember where was the last location
    fig.canvas.draw_idle()

def update(val):
    config = np.radians([s1.val, s2.val, s3.val])
    pts = joint_positions(config, h1=0)
    bones.set_data(pts[:, 0], pts[:, 1])
    fig.canvas.draw_idle()


def launch():
    setup_scene()
    draw_config(initial_config)
    plt.show()

