import plotly.graph_objects as go
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
import matplotlib.cm as cm
import random


def plot_UAV_FLPO(drone_START, drone_END, Facilities, figuresize=(6, 5)):

    facecolor = "#D3D3D3"
    # facecolor = "#FFFFFF"
    edgecolor= "#000000"
    start_locs = drone_START.squeeze(1).cpu().numpy()
    end_locs = drone_END.squeeze(1).cpu().numpy()
    f_locs = Facilities.squeeze(0).detach().cpu().numpy()

    plt.figure(figsize = figuresize, facecolor = facecolor,edgecolor= edgecolor)
    plt.scatter(start_locs[:, 0], start_locs[:, 1], color="purple", marker="X", alpha=0.2, s=200, label="S")
    plt.scatter(end_locs[:, 0], end_locs[:, 1], color="red", marker="*", label="E", s=200)
    plt.scatter(f_locs[:, 0], f_locs[:, 1], color="black", marker="^", label="F", s=100)
    plt.legend(facecolor = facecolor, edgecolor= edgecolor, fontsize=24)
    plt.axis('off')
    plt.show()


def plot_UAV_FLPO_3D(
    drone_START,
    drone_END,
    Facilities,
    start_altitudes=None,
    end_altitudes=None,
    facility_heights=None,
    scene_title="UAV Spatial Deployment",
    output_html=None,
):
    """
    3D interactive UAV start/end locations and service facilities.

    Parameters
    ----------
    drone_START : torch.Tensor or array-like, shape (B, 1, 2)
        XY start coordinates.
    drone_END   : torch.Tensor or array-like, shape (B, 1, 2)
        XY end   coordinates.
    Facilities  : torch.Tensor or array-like, shape (F, 2)
        XY facility coordinates.
    start_altitudes : array-like of shape (B,), optional
        Z heights for start points. Defaults to +10 units.
    end_altitudes   : array-like of shape (B,), optional
        Z heights for end   points. Defaults to +5 units.
    facility_heights: array-like of shape (F,), optional
        Z heights (building heights). Defaults to +2 units.
    scene_title    : str
        Title for the 3D scene.
    output_html    : str or None
        If provided, write the scene to this HTML file and return filepath.
    """

    # Convert tensors to numpy
    def to_np(x, se):
        if hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()
        return np.squeeze(x, axis=1) if se else np.squeeze(x, axis=0)

    start_xy = to_np(drone_START, 1)
    end_xy = to_np(drone_END, 1)
    fac_xy = to_np(Facilities, 0)

    B = start_xy.shape[0]
    F = fac_xy.shape[0]

    start_z = np.full(B, 1.0) if start_altitudes is None else np.array(start_altitudes)
    end_z = np.full(B, 0.0) if end_altitudes is None else np.array(end_altitudes)
    fac_z = np.full(F, 1.0) if facility_heights is None else np.array(facility_heights)

    fig = go.Figure()

    # Start locations (drones taking off)
    fig.add_trace(
        go.Scatter3d(
            x=start_xy[:, 0],
            y=start_xy[:, 1],
            z=start_z,
            mode="text",
            text=["𖥂"] * len(start_xy),
            textfont=dict(
                size=16, color="cyan"  # Adjust this value to change icon size
            ),
            name="Drone Start",
        )
    )

    # End locations (drones landing)
    fig.add_trace(
        go.Scatter3d(
            x=end_xy[:, 0],
            y=end_xy[:, 1],
            z=end_z,
            mode="text",
            text=["📍"] * len(end_xy),
            textfont=dict(
                size=20, color="red"  # Adjust this value to change icon size
            ),
            name="Drone End",
        )
    )

    # Facility elevation columns and top markers
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

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(
                title="X",
                backgroundcolor="rgb(10,10,20)",
                gridcolor="gray",
                zerolinecolor="gray",
            ),
            yaxis=dict(
                title="Y",
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
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.7)),
            aspectmode="data",
        ),
        title=dict(text=scene_title, font=dict(size=20), x=0.5),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="white"),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    if output_html:
        fig.write_html(output_html)
        return output_html
    else:
        fig.show()


def plot_uavFLPO_with_routes(
    START_locs, 
    END_locs, 
    F_locs, 
    final_cost,
    runtime,
    routes, 
    fraction_to_plot=0.3, 
    textcoords=[0.3, 0.95],
    textsize=18,
    random_seed=None
):
    """
    Plot a subset of UAV routes, color-coded by destination, with a clean visual design.
    """

    start_locs_np = START_locs.cpu().numpy().reshape(-1,2)
    end_locs_np = END_locs.cpu().numpy().reshape(-1,2)
    fac_locs = F_locs.detach().cpu().numpy().squeeze()

    num_uavs = start_locs_np.shape[0]
    num_facs = fac_locs.shape[0]
    num_to_plot = max(1, int(fraction_to_plot * num_uavs))

    if random_seed is not None:
        random.seed(random_seed)

    selected_indices = set(random.sample(range(num_uavs), num_to_plot))

    # Assign color by unique destination
    unique_dests, dest_indices = np.unique(end_locs_np, axis=0, return_inverse=True)
    dest_cmap = cm.get_cmap('tab10', len(unique_dests))

    plt.figure(figsize=(12, 8))

    # Plot all UAV start/end locations (transparent if not selected)
    for i in range(num_uavs):
        color = dest_cmap(dest_indices[i])
        alpha = 1.0 if i in selected_indices else 0.3
        plt.scatter(start_locs_np[i, 0], start_locs_np[i, 1], marker='s', c=[color], alpha=alpha, edgecolor='black', s=50)
        plt.scatter(end_locs_np[i, 0], end_locs_np[i, 1], marker='*', c=[color], alpha=alpha, edgecolor='black', s=400)

    # Plot selected UAV routes
    for i in selected_indices:
        color = dest_cmap(dest_indices[i])
        start = start_locs_np[i]
        end = end_locs_np[i]
        route_coords = fac_locs[routes[i]] if routes[i] else np.empty((0, 2))
        full_path = np.vstack([start, route_coords, end])
        plt.plot(full_path[:, 0], full_path[:, 1], '-o', color=color)

    # Plot all facility locations
    plt.scatter(fac_locs[:, 0], fac_locs[:, 1], c='gray', marker='.', s=50, alpha=0.4)

    # Create minimal legend
    legend_elements = [
        mlines.Line2D([], [], color='gray', marker='s', linestyle='None', markersize=8, label='UAV start'),
        mlines.Line2D([], [], color='gray', marker='X', linestyle='None', markersize=15, label='UAV end'),
        mlines.Line2D([], [], color='gray', marker='.', linestyle='None', markersize=12, label='Facility'),
        mlines.Line2D([], [], color='black', linestyle='-', label='Path')
    ]
    # plt.legend(handles=legend_elements, fontsize=24, ncol=4, loc='bottom center')

    # # add text for M, N, n_ends and final cost
    xcords, ycords = textcoords
    # plt.text(
    #     xcords,
    #     ycords,
    #     f"N={num_uavs}, M={num_facs},\nD={final_cost:.2e}, T={runtime:.2e} s",
    #     fontsize=textsize,
    #     color='black')

    # Add styled text with box
    plt.text(
        xcords,
        ycords,
        f"$N={num_uavs}$, $M={num_facs}$, $D={final_cost:.2f}$, $T={runtime:.2f}$ s",
        fontsize=textsize,
        color='black',
        fontname='Times New Roman', # 'DejaVu Serif',  Or 'Times New Roman'
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='round,pad=0.3',
            linewidth=1.0
        ),
        ha='left',  # horizontal alignment
        va='top'    # vertical alignment
    )


    # Clean plot: remove ticks and labels, keep frame
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    plt.box(True)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    drone_START = torch.rand(5, 1, 2) * 100
    drone_END = torch.rand(5, 1, 2) * 100
    Facilities = torch.rand(1, 3, 2) * 100

    # Call the function
    html_file = plot_UAV_FLPO_3D(
        drone_START,
        drone_END,
        Facilities,
        start_altitudes=np.linspace(12, 8, len(drone_START)),
        end_altitudes=np.linspace(6, 4, len(drone_END)),
        facility_heights=np.random.uniform(1.5, 3.5, size=Facilities.shape[1]),
        scene_title="UAV Deployment Overview",
        output_html="uav_scene.html",
    )
    print("Saved interactive figure to", html_file)

