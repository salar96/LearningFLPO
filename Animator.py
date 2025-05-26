import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import io
import logging
import warnings
import time

def animate_UAV_FLPO(
    drone_START,
    drone_END,
    Facilities_over_time,
    figuresize=(6, 5),
    interval=200,
    save_path="facility_movement.gif",
):
    facecolor = "#1a1a1a"
    edgecolor = "#08D3D6"
    start_locs = drone_START.squeeze(1).cpu().numpy()
    end_locs = drone_END.squeeze(1).cpu().numpy()
    # Facilities_over_time is a list of facility location arrays
    f_locs_list = [f.squeeze(0).detach().cpu().numpy() for f in Facilities_over_time]

    # Set up the figure
    fig = plt.figure(figsize=figuresize, facecolor=facecolor, edgecolor=edgecolor)
    ax = plt.axes()
    ax.set_facecolor(facecolor)

    # Plot static drone start and end locations
    ax.scatter(
        start_locs[:, 0],
        start_locs[:, 1],
        color="cyan",
        marker="o",
        alpha=0.5,
        label="S",
    )
    ax.scatter(end_locs[:, 0], end_locs[:, 1], color="red", marker="*", label="E")

    # Initialize facility scatter plot (to be updated in animation)
    f_scatter = ax.scatter([], [], color="gray", marker="^", label="F")
    # ax.legend(facecolor=facecolor, edgecolor=edgecolor, loc = 'upper left')
    ax.axis("off")

    # Set axis limits based on all data
    all_x = np.concatenate(
        [start_locs[:, 0], end_locs[:, 0]] + [f[:, 0] for f in f_locs_list]
    )
    all_y = np.concatenate(
        [start_locs[:, 1], end_locs[:, 1]] + [f[:, 1] for f in f_locs_list]
    )
    ax.set_xlim(min(all_x) - 0.1, max(all_x) + 0.1)
    ax.set_ylim(min(all_y) - 0.1, max(all_y) + 0.1)

    # Animation update function
    def update(frame):
        f_scatter.set_offsets(f_locs_list[frame])
        return (f_scatter,)

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(f_locs_list), interval=interval, blit=True
    )

    # Save the animation as GIF
    anim.save(save_path, writer="pillow")
    plt.close()

    return anim

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def animate_UAV_FLPO_3D(
    drone_START,
    drone_END,
    Facilities_list,
    start_altitudes=None,
    end_altitudes=None,
    facility_heights_list=None,
    scene_title="UAV Spatial Deployment Over Time",
    output_gif="facility_movement.gif",
    frame_duration=500,
    render_timeout=30,
    camera = (1.5,1.5,1.5)
):
    """
    Creates a GIF of 3D UAV start/end locations and facility movements over time.

    Parameters
    ----------
    drone_START : torch.Tensor or array-like, shape (B, 1, 2)
        XY start coordinates.
    drone_END   : torch.Tensor or array-like, shape (B, 1, 2)
        XY end coordinates.
    Facilities_list : list of torch.Tensor or array-like, each of shape (F, 2) or (1, F, 2)
        List of XY facility coordinates for each time step.
    start_altitudes : array-like of shape (B,), optional
        Z heights for start points. Defaults to +10 units.
    end_altitudes   : array-like of shape (B,), optional
        Z heights for end points. Defaults to +5 units.
    facility_heights_list : list of array-like of shape (F,), optional
        Z heights for facilities at each time step. Defaults to +2 units.
    scene_title    : str
        Title for the 3D scene.
    output_gif     : str
        Filepath for the output GIF.
    frame_duration : int
        Duration of each frame in milliseconds.
    render_timeout : int
        Timeout in seconds for rendering each frame.
    """

    # Convert tensors to numpy
    def to_np(x, se):
        if hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()
        if se:
            return np.squeeze(x, axis=1)  # For drone_START, drone_END
        else:
            # Handle Facilities shape (1, F, 2) -> (F, 2)
            x = np.squeeze(x, axis=0) if x.shape[0] == 1 else x
            if len(x.shape) == 3:
                logging.warning(f"Facilities tensor has shape {x.shape}; assuming batch dim=1")
                x = x[0]  # Take first batch if shape is (1, F, 2)
            elif len(x.shape) != 2:
                raise ValueError(f"Facilities tensor has invalid shape {x.shape}; expected (F, 2)")
            return x

    try:
        start_xy = to_np(drone_START, 1)
        end_xy = to_np(drone_END, 1)
        fac_xy_list = [to_np(fac, 0) for fac in Facilities_list]
    except Exception as e:
        logging.error(f"Error converting inputs to numpy: {e}")
        raise

    B = start_xy.shape[0]
    F = fac_xy_list[0].shape[0] if fac_xy_list else 0
    T = len(Facilities_list)
    c_x , c_y , c_z = camera
    if T == 0:
        logging.error("Facilities_list is empty.")
        raise ValueError("Facilities_list must contain at least one time step.")

    start_z = np.full(B, 10.0) if start_altitudes is None else np.array(start_altitudes)
    end_z = np.full(B, 5.0) if end_altitudes is None else np.array(end_altitudes)
    fac_z_list = (
        [np.full(F, 2.0) for _ in range(T)] if facility_heights_list is None
        else [np.array(fh) for fh in facility_heights_list]
    )


    # Create figure
    fig = go.Figure()

    # Define frames
    frames = []
    for t in range(T):
        frame_traces = []

        # Start locations (drones taking off)
        frame_traces.append(
            go.Scatter3d(
                x=start_xy[:, 0],
                y=start_xy[:, 1],
                z=start_z,
                mode="text",
                text=["𖥂"] * len(start_xy),
                textfont=dict(size=16, color="cyan"),
                name="Drone Start",
            )
        )

        # End locations (drones landing)
        frame_traces.append(
            go.Scatter3d(
                x=end_xy[:, 0],
                y=end_xy[:, 1],
                z=end_z,
                mode="text",
                text=["📍"] * len(end_xy),
                textfont=dict(size=20, color="red"),
                name="Drone End",
            )
        )

        # Facility elevation columns and top markers
        fac_xy = fac_xy_list[t]
        fac_z = fac_z_list[t]
        for xi, yi, zi in zip(fac_xy[:, 0], fac_xy[:, 1], fac_z):
            frame_traces.append(
                go.Scatter3d(
                    x=[xi, xi],
                    y=[yi, yi],
                    z=[0, zi],
                    mode="lines",
                    line=dict(color="gray", width=3),
                    showlegend=False,
                )
            )

        frame_traces.append(
            go.Scatter3d(
                x=fac_xy[:, 0],
                y=fac_xy[:, 1],
                z=fac_z,
                mode="markers",
                marker=dict(size=5, symbol="square", color="silver"),
                name=f"Facilities (t={t})",
            )
        )

        frames.append(go.Frame(data=frame_traces, name=f"frame{t}"))

    # Add initial traces (first time step)
    fac_xy = fac_xy_list[0]
    fac_z = fac_z_list[0]
    fig.add_trace(
        go.Scatter3d(
            x=start_xy[:, 0],
            y=start_xy[:, 1],
            z=start_z,
            mode="text",
            text=["𖥂"] * len(start_xy),
            textfont=dict(size=16, color="cyan"),
            name="Drone Start",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=end_xy[:, 0],
            y=end_xy[:, 1],
            z=end_z,
            mode="text",
            text=["📍"] * len(end_xy),
            textfont=dict(size=20, color="red"),
            name="Drone End",
        )
    )
    for xi, yi, zi in zip(fac_xy[:, 0], fac_xy[:, 1], fac_z):
        fig.add_trace(
            go.Scatter3d(
                x=[xi, xi],
                y=[yi, yi],
                z=[0, zi],
                mode="lines",
                line=dict(color="gray", width=3),
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=fac_xy[:, 0],
            y=fac_xy[:, 1],
            z=fac_z,
            mode="markers",
            marker=dict(size=5, symbol="square", color="silver"),
            name="Facilities",
        )
    )

    # Simplified layout to reduce rendering load
    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(
                title="X",
                range = [0, 1],
                dtick=0.1,
                backgroundcolor="rgb(10,10,20)",
                gridcolor="gray",
                zerolinecolor="gray",
            ),
            yaxis=dict(
                title="Y",
                range = [0, 1],
                dtick=0.1,
                backgroundcolor="rgb(10,10,20)",
                gridcolor="gray",
                zerolinecolor="gray",
            ),
            zaxis=dict(
                title="Z",
                backgroundcolor="rgb(10,10,20)",
                gridcolor="gray",
                zerolinecolor="gray",
            ),
            camera=dict(eye=dict(x=c_x, y=c_y, z=c_z)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        title=dict(text=scene_title, font=dict(size=20), x=0.5),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="white"),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    
    # Add frames to figure
    fig.frames = frames

    # Generate images for each frame
    images = []
    for t in range(T):
        logging.info(f"Rendering frame {t}/{T-1}")
        start_time = time.time()
        try:
            temp_fig = go.Figure(data=frames[t].data, layout=fig.layout)
            # Lower resolution to reduce memory usage
            img_bytes = temp_fig.to_image(format="png", engine="kaleido")
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)
        except Exception as e:
            logging.error(f"Failed to render frame {t}: {e}")
            warnings.warn(f"Skipping frame {t} due to rendering error: {e}")
            continue
        if time.time() - start_time > render_timeout:
            logging.warning(f"Frame {t} took too long to render (> {render_timeout}s); skipping")
            continue

    if not images:
        logging.error("No frames were successfully rendered.")
        raise RuntimeError("Failed to render any frames for the GIF.")

    # Save GIF
    logging.info(f"Saving GIF to {output_gif}")
    try:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=frame_duration,
            loop=0
        )
    except Exception as e:
        logging.error(f"Failed to save GIF: {e}")
        raise

    logging.info(f"GIF successfully saved to {output_gif}")
    return output_gif

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    drone_START = torch.rand(5, 1, 2)
    drone_END = torch.rand(5, 1, 2)
    Facilities = [torch.rand(1, 3, 2) for _ in range(2)]
    # Call the function
    animate_UAV_FLPO_3D(
        drone_START,
        drone_END,
        Facilities,
        start_altitudes=None,
        end_altitudes=None,
        facility_heights_list=None,
        scene_title="UAV Spatial Deployment Over Time",
        output_gif="facility_movement.gif",
        frame_duration=500,
        render_timeout=30,
        camera = (1.5,1.5,0.5)
    )
