import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from kinematics import joint_positions
from planning import choose_trajectory
from config import D2, D3, D4, H2, H3, MAX_LIFT

STEPS = 10
THRESHOLD = 0.1

REACH = D2 + D3 + D4
Z_TOP = MAX_LIFT + H2 + H3

fig = None
ax = None
bones = None
carriage = None
s_theta1 = None
s_theta2 = None
s_theta3 = None
s_h1 = None

# Initialize global state variables
current_config = (0.0, np.pi, -np.pi, 0.0) # (theta1, theta2, theta3, h1)
anim = None
trail = None
target_marker = None

def setup_scene():
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection="3d")

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.26, top=1.0)

    ax.set_xlim(-REACH * 1.1, REACH * 1.1)
    ax.set_ylim(-REACH * 1.1, REACH * 1.1)
    ax.set_zlim(0.0, Z_TOP * 1.1)

    ax.set_box_aspect((2 * REACH * 1.1, 2 * REACH * 1.1, Z_TOP * 1.1))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
   
    ax.plot([0, 0], [0, 0], [0, MAX_LIFT],
            color="0.6", linewidth=2, linestyle="--")
    
    bones, = ax.plot([], [], [], "-o", linewidth=3, markersize=6, color="C0")
    carriage, = ax.plot([], [], [], "s", markersize=12, color="C3")
    target_marker, = ax.plot([], [], [], "rx", markersize=12, markeredgewidth=2)
    trail, = ax.plot([], [], [], ".", markersize=2, alpha=0.6)

    return fig, ax, bones, carriage, target_marker, trail

def draw_config(config):
    pts = joint_positions(config[:3], config[3])   # (4, 3): x, y, z

    bones.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
    carriage.set_data_3d([0], [0], [config[3]])    # carriage sits at z = h1

    fig.canvas.draw_idle()

def on_slider(val):
    config = (
        *np.radians([s_theta1.val, s_theta2.val, s_theta3.val]),
        s_h1.val,                                  # h1 is a length, NOT an angle
    )
    draw_config(config)

def on_click(event):
    global current_config, anim
    
    # 3D axis
    if event.inaxes != ax:
        return
    
    # Extract 2D projection click and use h1 for Z
    target_x, target_y = event.xdata, event.ydata
    current_z = current_config[3] + H2 + H3
    
    target_marker.set_data_3d([target_x], [target_y], [current_z])
    
    gamma = np.arctan2(target_y, target_x)
    target_pose = (target_x, target_y, current_z, gamma)
    
    trajectory, mode = choose_trajectory(current_config, target_pose, STEPS, THRESHOLD)
    ax.set_title(f"mode: {mode}")

    if trajectory is None:
        fig.canvas.draw_idle()
        return 
    
    # Clear the previous 3D trail
    trail.set_data_3d([], [], []) 

    def frame(i):
        config = trajectory[i]
        draw_config(config)
        
        # Calculate 3D tip position using the config's h1
        tip = joint_positions(config[:3], config[3])[-1] 
        
        # Grow the 3D trail
        xs, ys, zs = trail.get_data_3d()
        trail.set_data_3d([*xs, tip[0]], [*ys, tip[1]], [*zs, tip[2]])
        return bones, trail
    
    anim = FuncAnimation(fig, frame, frames=len(trajectory),
                         interval=20, blit=False, repeat=False)
    
    # Remember the last location
    current_config = tuple(trajectory[-1])  
    
    # Sync the sliders to the new position
    for s in (s_theta1, s_theta2, s_theta3, s_h1):
        s.disconnect_events()
        
    s_theta1.set_val(np.degrees(current_config[0]))
    s_theta2.set_val(np.degrees(current_config[1]))
    s_theta3.set_val(np.degrees(current_config[2]))
    s_h1.set_val(current_config[3])
    
    for s in (s_theta1, s_theta2, s_theta3, s_h1):
        s.on_changed(on_slider)

    fig.canvas.draw_idle()


def launch():
    global fig, ax, bones, carriage, target_marker, trail
    global s_theta1, s_theta2, s_theta3, s_h1

    fig, ax, bones, carriage, target_marker, trail = setup_scene()

    # slider axes, stacked along the bottom
    ax_t1 = fig.add_axes([0.25, 0.20, 0.55, 0.03])
    ax_t2 = fig.add_axes([0.25, 0.15, 0.55, 0.03])
    ax_t3 = fig.add_axes([0.25, 0.10, 0.55, 0.03])
    ax_h1 = fig.add_axes([0.25, 0.05, 0.55, 0.03])

    s_theta1 = Slider(ax_t1, "θ1 (deg)", -180, 180, valinit=0)
    s_theta2 = Slider(ax_t2, "θ2 (deg)", -180, 180, valinit=180)
    s_theta3 = Slider(ax_t3, "θ3 (deg)", -180, 180, valinit=-180)
    s_h1     = Slider(ax_h1, "h1 (lift)", 0, MAX_LIFT, valinit=MAX_LIFT * 0.4)

    for s in (s_theta1, s_theta2, s_theta3, s_h1):
        s.on_changed(on_slider)

    fig.canvas.mpl_connect('button_press_event', on_click)
    on_slider(None)   # draw the initial pose once

    plt.show()
