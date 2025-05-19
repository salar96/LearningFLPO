import torch
import matplotlib.pyplot as plt
import numpy as np


def RelevanceMaskGenerator(points, start_indices):
    """
    Creates a binary mask for each batch of 2D points based on two geometric conditions.

    Parameters:
    - points (torch.Tensor): Tensor of shape (B, N, 2) representing B batches of N 2D points.
    - start_indices (torch.Tensor): Tensor of shape (B,) representing the index of the start point in each batch.

    Returns:
    - mask (torch.Tensor): Binary mask of shape (B, N) where each element is 1 if the corresponding point satisfies either:
        a) Squared distance from the start point is >= distance from start to end.
        b) Lies on or on the side of the separating line (perpendicular to the start-end line at the start point) that does not include the end point.
    """

    B, N, _ = points.shape

    # Step 1: Gather start points based on start_indices
    start_points = points.gather(
        1, start_indices.long().view(B, 1, 1).expand(B, 1, 2)
    ).squeeze(1)

    # Step 2: Get end points (last point in each batch)
    end_points = points[:, -1, :]

    # Step 3: Compute vector from start to end
    v = end_points - start_points  # Shape: (B, 2)

    # Step 4: Compute squared Euclidean distance from start to end
    start_to_end_sq = (v**2).sum(dim=1, keepdim=True)  # Shape: (B, 1)

    # Step 5: Compute squared distances from all points to start
    diff = points - start_points.unsqueeze(1)  # Shape: (B, N, 2)
    squared_dist = (diff**2).sum(dim=2)  # Shape: (B, N)

    # Step 6: Compute dot products of (point - start) with vector v
    dot_products = (diff * v.unsqueeze(1)).sum(dim=2)  # Shape: (B, N)

    # Step 7: Apply conditions
    cond_a = squared_dist > start_to_end_sq  # Shape: (B, N)
    cond_b = dot_products <= 0  # Shape: (B, N)

    # Step 8: Combine conditions with logical OR
    mask = (cond_a | cond_b) #.float()  # Shape: (B, N)

    return mask


if __name__ == "__main__":
    from utils import *
    # Generate random data
    B, N = 1, 50
    points = torch.rand(B, N, 2)
    points = generate_unit_circle_cities(B, N, 2)
    start_indices = torch.randint(0, N, (B,))

    # Compute mask
    mask = RelevanceMaskGenerator(points, start_indices)

    # Convert to numpy for plotting
    points_np = points[0].numpy()
    mask_np = mask[0].numpy().astype(bool)

    start_idx = start_indices[0].item()
    start_point = points_np[start_idx]
    end_point = points_np[-1]

    # Plot all points
    plt.figure(figsize=(10, 8))
    plt.scatter(
        points_np[:, 0], points_np[:, 1], c="gray", alpha=0.3, label="All Points"
    )

    # Highlight masked points
    plt.scatter(
        points_np[mask_np, 0], points_np[mask_np, 1], c="red", label="Masked Points"
    )

    # Highlight start and end points
    plt.scatter(*start_point, c="green", s=100, marker="s", label="Start Point")
    plt.scatter(*end_point, c="purple", s=100, marker="d", label="End Point")

    # Draw line between start and end
    plt.plot(
        [start_point[0], end_point[0]],
        [start_point[1], end_point[1]],
        "k--",
        label="Start → End",
    )

    # Draw perpendicular line at start (separating line)
    v = end_point - start_point
    perp_dir = np.array([-v[1], v[0]])  # Perpendicular direction
    scale = 1.5
    p1 = start_point + scale * perp_dir
    p2 = start_point - scale * perp_dir
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "m--", alpha=0.7, label="Separating Line")

    # Formatting
    plt.legend()
    plt.title("Binary Mask Based on Geometric Conditions")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
