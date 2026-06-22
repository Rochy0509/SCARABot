import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from kinematics import joint_positions, forward_kinematics
from planning import choose_trajectory
from config import D2, D3, D4, H2, H3

REACH = D2 + D3 + D4 #arm's max reach
STEPS = 10 #frames per move
THRESHOLD = 0.1 #max joint jump before cartesian planning is rejected
Z_FIXED = H2 + H3 #lift at h1 = 0 for 2D demo
GAMMA_FIXED = 0.0 #tool orientation fixed

s_theta1 = None
s_theta2 = None
s_theta3 = None

fig = None
ax = None
bones = None

anim = None
trail = None
target_marker = None

#state
current_config = (*np.radians([0, 180, -180]), 0.0)


def setup_scene():
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

    return fig, ax, bones, target_marker, trail


def draw_config(config):
    pts = joint_positions(config[:3], h1=0)
    bones.set_data(pts[:, 0], pts[:, 1])

    fig.canvas.draw_idle()

def on_slider(val):
    config = ( *np.radians([s_theta1.val, s_theta2.val, s_theta3.val]), 0.0)
    draw_config(config)


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

    for s in (s_theta1, s_theta2, s_theta3):
        s.disconnect_events()
        
    s_theta1.set_val(np.degrees(current_config[0]))
    s_theta2.set_val(np.degrees(current_config[1]))
    s_theta3.set_val(np.degrees(current_config[2]))
    
    for s in (s_theta1, s_theta2, s_theta3):
        s.on_changed(on_slider)

    fig.canvas.draw_idle()
   

def update(val):
    config = np.radians([s1.val, s2.val, s3.val])
    pts = joint_positions(config, h1=0)
    bones.set_data(pts[:, 0], pts[:, 1])
    fig.canvas.draw_idle()


def launch():
    global fig, ax, bones, target_marker, trail
    global s_theta1, s_theta2, s_theta3

    fig, ax, bones, target_marker, trail = setup_scene()

    #sliders
    ax_t1 = fig.add_axes([0.25, 0.18, 0.55, 0.03])
    ax_t2 = fig.add_axes([0.25, 0.12, 0.55, 0.03])
    ax_t3 = fig.add_axes([0.25, 0.06, 0.55, 0.03])

    s_theta1 = Slider(ax_t1, "theta1", -180, 180, valinit=0)
    s_theta2 = Slider(ax_t2, "theta2", -180, 180, valinit=180)
    s_theta3 = Slider(ax_t3, "theta3", -180, 180, valinit=-180)

    for s in (s_theta1, s_theta2, s_theta3):
        s.on_changed(on_slider)

    fig.canvas.mpl_connect('button_press_event', on_click)
    on_slider(None)   # draw the initial pose once

    plt.show()

