import torch
import time
import numpy as np

def sa_gpu(
    s,
    e,
    num_nodes,
    num_agents,
    dim,
    Y_init=None,
    YMIN=0.0,
    YMAX=1.0,
    iters=1000,
    T_start=1.0,
    T_end=1e-3,
    alpha=0.995,
    verbose=False,
):
    T = num_nodes + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def initialize_solution():
        if Y_init is not None:
            y = torch.tensor(Y_init.copy().reshape(num_nodes, dim), device=device)
        else:
            y = torch.rand((num_nodes, dim), device=device) * (YMAX - YMIN) + YMIN

        eta = torch.zeros((num_agents, T, num_nodes + 1), dtype=torch.int32, device=device)
        for a in range(num_agents):
            for t in range(T - 1):
                j = torch.randint(0, num_nodes, (1,)).item()
                eta[a, t, j] = 1
            eta[a, T - 1, num_nodes] = 1  # final step to end
        return y, eta

    def evaluate_cost(y, eta):
        total = 0.0
        for a in range(num_agents):
            prev = s[a, 0]
            for t in range(T):
                j = torch.argmax(eta[a, t])
                curr = y[j] if j < num_nodes else e[a, 0]
                total += torch.sum((curr - prev) ** 2)
                prev = curr
        return total

    def perturb_solution(y, eta, y_sigma=0.1, eta_prob=0.05):
        new_y = y + torch.randn_like(y) * y_sigma
        new_y = torch.clamp(new_y, YMIN, YMAX)
        new_eta = eta.clone()
        for a in range(num_agents):
            for t in range(T - 1):
                if torch.rand(1).item() < eta_prob:
                    new_eta[a, t] = 0
                    j = torch.randint(0, num_nodes, (1,)).item()
                    new_eta[a, t, j] = 1
        return new_y, new_eta

    start_time = time.time()
    y, eta = initialize_solution()
    best_y, best_eta = y.clone(), eta.clone()
    best_cost = evaluate_cost(y, eta)
    cost_history = [best_cost]

    T_curr = T_start
    for i in range(iters):
        y_new, eta_new = perturb_solution(y, eta)
        cost_new = evaluate_cost(y_new, eta_new)
        delta = cost_new - best_cost
    

        if delta < 0 or torch.rand(1, device=device).item() < torch.exp(-delta / T_curr).item():

            y, eta = y_new, eta_new
            if cost_new < best_cost:
                best_cost = cost_new
                best_y = y_new.clone()
                best_eta = eta_new.clone()
        cost_history.append(best_cost)
        T_curr *= alpha
        if verbose and (i % max(1, iters // 10) == 0 or i == iters - 1):
            print(
                f"Iteration {i+1}/{iters}, Best Cost: {best_cost:.4f}, Temperature: {T_curr:.4f}"
            )

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Elapsed time: {elapsed_time:.2f} seconds Best Cost: {best_cost:.4f}")

    return (
        best_y.detach().cpu().numpy(),
        best_eta.detach().cpu().numpy(),
        best_cost,
        elapsed_time,
    )

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
    num_agents = 20
    dim = 2
    # Problem setup
    s = torch.rand((num_agents, 1, dim), dtype=torch.float32, device=device)
    e = torch.rand((num_agents, 1, dim), dtype=torch.float32, device=device)
    

    # Run GA
    with torch.no_grad():
        best_y, best_eta, best_cost, elapsed_time = sa_gpu(
            s,
            e,
            num_nodes,
            num_agents,
            dim,
        )