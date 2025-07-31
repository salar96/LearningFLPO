import torch
import time

def cem_gpu(
    s,
    e,
    num_nodes,
    num_agents,
    dim,
    Y_init=None,
    YMIN=0.0,
    YMAX=1.0,
    n_iter=100,
    pop_size=100,
    elite_frac=0.2,
    verbose=False,
    device="cuda",
):
    T = num_nodes + 1

    elite_size = int(pop_size * elite_frac)

    if Y_init is not None:
        y_mean = torch.tensor(Y_init.copy().reshape(num_nodes, dim), device=device)
    else:
        y_mean = (YMAX - YMIN) * torch.rand((num_nodes, dim), device=device) + YMIN
    y_std = torch.ones((num_nodes, dim), device=device) * 0.5

    eta_logits = torch.randn(num_agents, T, num_nodes + 1, device=device)

    best_cost = float("inf")
    best_y = None
    best_eta = None

    def evaluate_batch_cost(y_batch, eta_batch):
        total_cost = torch.zeros(pop_size, device=device)
        for k in range(pop_size):
            for a in range(num_agents):
                prev = s[a]
                for t in range(T):
                    j = torch.argmax(eta_batch[k, a, t]).item()
                    curr = y_batch[k, j] if j < num_nodes else e[a]
                    total_cost[k] += torch.sum((curr - prev) ** 2)
                    prev = curr
        return total_cost

    start_time = time.time()

    for iteration in range(n_iter):
        # Sample y
        eps = torch.randn((pop_size, num_nodes, dim), device=device)
        y_samples = y_mean.unsqueeze(0) + y_std.unsqueeze(0) * eps
        y_samples = y_samples.clamp(YMIN, YMAX)

        # Sample eta
        eta_samples = torch.zeros((pop_size, num_agents, T, num_nodes + 1), device=device)
        for a in range(num_agents):
            for t in range(T):
                probs = torch.softmax(eta_logits[a, t], dim=-1)
                d = torch.distributions.Categorical(probs)
                sampled = d.sample((pop_size,))
                one_hot = torch.nn.functional.one_hot(sampled, num_nodes + 1).float()
                eta_samples[:, a, t, :] = one_hot

        # Force last step to go to END
        eta_samples[:, :, -1, :] = 0
        eta_samples[:, :, -1, num_nodes] = 1

        # Evaluate cost
        costs = evaluate_batch_cost(y_samples, eta_samples)

        # Update best
        min_cost, min_idx = torch.min(costs, dim=0)
        if min_cost.item() < best_cost:
            best_cost = min_cost.item()
            best_y = y_samples[min_idx].detach().cpu().numpy()
            best_eta = eta_samples[min_idx].detach().cpu().numpy()

        # Select elite
        elite_idxs = torch.topk(costs, elite_size, largest=False).indices
        elite_y = y_samples[elite_idxs]
        y_mean = elite_y.mean(dim=0)
        y_std = elite_y.std(dim=0) + 1e-6

        # Update logits
        for a in range(num_agents):
            for t in range(T):
                elite_eta = eta_samples[elite_idxs, a, t, :]
                avg = elite_eta.mean(dim=0)
                eta_logits[a, t] = torch.log(avg + 1e-6)

        if verbose and (iteration % max(1, n_iter // 10) == 0 or iteration == n_iter - 1):
            print(f"Iteration {iteration+1}/{n_iter}, Best Cost: {best_cost:.4f}")

    elapsed_time = time.time() - start_time
    return best_y, best_eta, best_cost, elapsed_time

if __name__ == "__main__":
    # Example usage
    import torch
    import numpy as np

    # Set seed for reproducibility
    np.random.seed(150)
    torch.manual_seed(150)

    # Move input to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = 4
    num_agents = 2
    dim = 2
    # Problem setup
    s = torch.rand((num_agents, 1, dim), dtype=torch.float32, device=device)
    e = torch.rand((num_agents, 1, dim), dtype=torch.float32, device=device)
    

    # Run GA
    with torch.no_grad():
        best_y, best_eta, best_cost, elapsed_time = cem_gpu(
            s,
            e,
            num_nodes,
            num_agents,
            dim,
        )