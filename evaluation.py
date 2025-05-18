
import torch
from VRP_Net_L import VRPNet_L
from VRP_Net_L_DAD import VRPNet_L_DAD
from VRP_Net_H import VRPNet_H
from matplotlib import pyplot as plt
from utils import *
import os
from BeamSearch import beam_search
import matplotlib.pyplot as plt
from torchinfo import summary
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patheffects import withStroke
from RelevanceMaskGenerator import RelevanceMaskGenerator
from inference import inference
def plot_routes(cities, routes, action_labels, save_folder='plots_TF'):
    """
    Plots the routes for a range of indices and saves the plots in a folder.

    Args:
        cities (torch.Tensor): Tensor of shape (B, N, 2) representing city coordinates.
        routes (torch.Tensor): Tensor of shape (B, N) representing routes.
        index_range (range): Range of indices to plot.
        save_folder (str): Folder to save the plots. Default is 'plots'.
    """
    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    for batch_index in torch.arange(len(data)):
        # Extract the specific batch
        cities_batch = cities[batch_index].numpy()
        route_batch = routes[batch_index].long().squeeze().numpy()
        true_route_batch = action_labels[batch_index]
        # Get coordinates of cities in the order of the route
        ordered_cities = cities_batch[route_batch]
        ordered_cities_true = cities_batch[true_route_batch]
        # Plot cities
        plt.figure(figsize=(8, 6))
        plt.scatter(cities_batch[1:-1, 0], cities_batch[1:-1, 1], marker='.', color='blue', zorder=2, label='Cities')
        # for i, (x, y) in enumerate(cities_batch):
        #     if i == 0:
        #         plt.text(x, y, 'S', fontsize=12, ha='right', color='black')
        #     elif i == index_range[-1]:
        #         plt.text(x, y, 'E', fontsize=12, ha='right', color='black')
        #     else:
        #         pass

        # Plot the route
        # plt.text(0.0, 0.0, str(cities[batch_index]), fontsize=12, ha='right', color='black')
        plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], color='red', linestyle='-', zorder=1, label='Route')
        plt.plot(ordered_cities_true[:, 0], ordered_cities_true[:, 1], color='green', linestyle='--', zorder=1, alpha=0.6,label='True Route')
        # Highlight start and end points
        plt.scatter(cities_batch[0, 0], cities_batch[0, 1], color='green', s=50, label='Start', zorder=3)
        plt.scatter(cities_batch[-1, 0], cities_batch[-1, 1], marker = "^",color='red', s=50, label='End', zorder=3)
        model_cost = route_cost(cities[batch_index:batch_index+1],routes[batch_index:batch_index+1])[0] #/straight_costs(cities[batch_index:batch_index+1])[0]
        true_cost = route_cost(cities[batch_index:batch_index+1],action_labels[batch_index:batch_index+1])[0]
        plt.title(f"Batch {batch_index} cost: {model_cost:.2f} True cost: {true_cost:.2f}") 
        
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.axis('off')

        # Save the plot to the specified folder
        save_path = os.path.join(save_folder, f"route_batch_{batch_index}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to avoid showing it

    print(f"Plots saved in '{save_folder}'")



def plot_routes_el(cities, routes, action_labels, save_folder='plots_TF', dpi=300):
    os.makedirs(save_folder, exist_ok=True)
    
    # Custom colormap for route gradient
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(routes[0])))
    
    for batch_idx in range(len(cities)):
        # Extract data
        city_coords = cities[batch_idx].numpy()
        route = routes[batch_idx].long().squeeze().numpy()
        true_route = action_labels[batch_idx]
        
        # Create figure with dark theme
        plt.figure(figsize=(10, 8),facecolor='#1a1a1a')# facecolor='#1a1a1a')
        ax = plt.subplot(111,facecolor='#1a1a1a')# facecolor='#1a1a1a')
        
        # Add subtle grid
        ax.grid(True, alpha=0.01, color='white', linestyle='--', linewidth=0.5)
        
        # Plot network connections (faint)
        for i in range(len(city_coords)):
            plt.plot([city_coords[i,0], city_coords[(i+1)%len(city_coords),0]],
                     [city_coords[i,1], city_coords[(i+1)%len(city_coords),1]],
                     color='gray', alpha=0.35, lw=0.5, zorder=1)

        # Plot true route with transparency
        true_segments = np.array([city_coords[true_route[:-1]], city_coords[true_route[1:]]])
        true_segments = np.transpose(true_segments, (1, 0, 2))
        true_lc = LineCollection(true_segments, color='cyan', alpha=0.3, 
                                 linewidth=3, linestyle='--', label='Optimal Route')
        ax.add_collection(true_lc)

        # Plot model route with gradient
        segments = np.array([city_coords[route[:-1]], city_coords[route[1:]]])
        segments = np.transpose(segments, (1, 0, 2))
        lc = LineCollection(segments, colors=colors, linewidth=4, 
                           path_effects=[withStroke(linewidth=6, foreground='black')], 
                           label='Model Route')
        ax.add_collection(lc)

        # City styling
        plt.scatter(city_coords[1:-1, 0], city_coords[1:-1, 1], 
                    c='#40a5ff', s=80, edgecolor='w', linewidth=0.5,
                    label='Intermediate Cities', zorder=3, marker='o')
        
        # Start/End markers
        plt.scatter(city_coords[0, 0], city_coords[0, 1], 
                    c='#22ff22', s=200, edgecolor='w', linewidth=1, 
                    marker='D', label='Start', zorder=4)
        plt.scatter(city_coords[-1, 0], city_coords[-1, 1], 
                    c='#ff4444', s=200, edgecolor='w', linewidth=1, 
                    marker='^', label='End', zorder=4)

        # Cost comparison
        model_cost = route_cost(cities[batch_idx:batch_idx+1], routes[batch_idx:batch_idx+1])[0]
        true_cost = route_cost(cities[batch_idx:batch_idx+1], action_labels[batch_idx:batch_idx+1])[0]
        cost_ratio = (model_cost / true_cost - 1) * 100
        
        # Custom title with gradient
        title_text = f'Route Optimization Analysis (Batch {batch_idx})\n'
        subtitle_text = f'Model: {model_cost:.2f} | Optimal: {true_cost:.2f} | Δ {cost_ratio:+.1f}%' #  | Δ {cost_ratio:+.1f}%
        
        plt.title(title_text + subtitle_text, color='white', fontsize=14, pad=20,
                  fontdict={'family': 'Arial', 'weight': 'bold'},
                  bbox=dict(facecolor='#333333', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))

        # Add legend with shadow
        legend = plt.legend(frameon=True, framealpha=0.9, edgecolor='none', 
                           loc='upper left', bbox_to_anchor=(1, 1), 
                           facecolor='#2d2d2d', labelcolor='white')
        legend.get_frame().set_linewidth(0)
        
        # Add watermark
        # plt.text(0.5, 0.5, 'Confidential', color='gray', alpha=0.2,
        #          fontsize=40, ha='center', va='center', rotation=30,
        #          transform=ax.transAxes)
        
        # Style adjustments
        plt.axis('equal')
        plt.tick_params(axis='both', colors='white', labelsize=8)
        plt.tight_layout()
        
        # Save with high quality
        save_path = os.path.join(save_folder, f"route_batch_{batch_idx}.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    print(f"🚀 High-impact visualizations saved to: '{save_folder}'")

if __name__ == "__main__":
    
    num_data = 5
    num_cities = 50
    city_dim = 2
    mod = 'eval_greedy'
    num_samples = 500

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Running on: " , device)


    model_classes = {'VRPNet_L': VRPNet_L}
    model = load_model('Saved_models/VRPNet_L_lr1e-04_bs32_ep60000_samples1920000_cities50_inputdim2_workers0_hidden64_enc1_dec1_heads8_dropout0.30_train_PO_2025_05_17_22_43_32last_model.pth', model_classes, device=device)

    print('model loaded')
    
    data = torch.rand(num_data,num_cities,city_dim).to(device)
    # data = USA_data.to(device)
    # data = generate_unit_circle_cities(num_data,num_cities,city_dim).to(device)
    num_data , num_cities , city_dim = data.shape

    with torch.no_grad():
        
        if mod == 'eval_greedy':
            _ , actions = inference(data, model , "Greedy")
                
        elif mod == 'BS':
            seq, scores = beam_search(model,data,beam_width=10)
            actions = seq[:,0,:]
        elif mod == 'eval_sampling':
            _ , actions = inference(data, model , "sampling")
    
    _,action_labels = generate_true_labels(data,beta=1e8)
    plot_routes_el(data.cpu(),actions.cpu(),action_labels)
    print(actions)
  
    
