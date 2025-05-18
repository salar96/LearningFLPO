import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
import matplotlib

def plot_UAV_FLPO(
    drone_START, drone_END, Facilities,
    figuresize=(8, 6),
    annotate=True,
    save_path=None,
    show_map=False,
    map_zoom=12,
    return_fig=False
):
    # Convert tensors to NumPy
    start_locs = drone_START.squeeze(1).cpu().numpy()
    end_locs = drone_END.squeeze(1).cpu().numpy()
    f_locs = Facilities.squeeze(0).detach().cpu().numpy()

    B = start_locs.shape[0]  # Number of drones

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figuresize)

    # Get a qualitative colormap
    cmap = matplotlib.colormaps['tab10'] if B <= 10 else matplotlib.colormaps['tab20']

    colors = [cmap(i % cmap.N) for i in range(B)]

    # Plot start and end points for each drone in unique colors
    for i in range(B):
        ax.scatter(*start_locs[i], color=colors[i], marker='X', s=100,
                   edgecolor='black', linewidth=0.5, label=f'Drone {i} Start')
        ax.scatter(*end_locs[i], color=colors[i], marker='o', s=50,
                   edgecolor='black', linewidth=0.5, label=f'Drone {i} End')

        if annotate:
            ax.text(*start_locs[i], f'{i}S', fontsize=8, ha='right', va='bottom',
                    path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])
            ax.text(*end_locs[i], f'{i}E', fontsize=8, ha='left', va='top',
                    path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])

    # Plot facilities
    ax.scatter(f_locs[:, 0], f_locs[:, 1], color='black', marker='^', s=90,
               edgecolor='white', linewidth=0.7, label='Facility')
    if annotate:
        for i, loc in enumerate(f_locs):
            txt = ax.text(loc[0], loc[1], f'F{i}', fontsize=9, weight='bold', ha='center', va='center', color='black')
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])

    # Add basemap using a valid tile provider
    if show_map:
        import contextily as ctx
        import geopandas as gpd
        from shapely.geometry import Point

        # Combine all points for map bounds
        all_coords = np.vstack([start_locs, end_locs, f_locs])
        points = [Point(xy) for xy in all_coords]
        gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326").to_crs(epsg=3857)

        # Plot invisible points to force extent
        gdf.plot(ax=ax, alpha=0)

        # Use CartoDB Positron instead of Stamen
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=map_zoom)
        ax.set_axis_off()
    else:
        ax.set_facecolor("#f9f9f9")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

    ax.set_title("UAV Start-End Locations with Facilities", fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=8, frameon=True)

    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', format=save_path.split('.')[-1])
    plt.show()
    if return_fig:
        return fig
    


if __name__ == "__main__":
    torch.manual_seed(0)
    drone_START = torch.rand(5, 1, 2) * 100
    drone_END = torch.rand(5, 1, 2) * 100
    Facilities = torch.rand(1, 3, 2) * 100
    print(matplotlib.get_backend())
    # Call the function
    plot_UAV_FLPO(drone_START, drone_END, Facilities,
                annotate=True,
                show_map=True,  # Set to False if offline
                return_fig=False)