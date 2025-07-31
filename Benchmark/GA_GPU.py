import torch
import time
import random


def ga_gpu(
    s,
    e,
    num_nodes,
    num_agents,
    dim,
    Y_init=None,
    YMIN=0.0,
    YMAX=1.0,
    pop_size=100,
    generations=50,
    mutation_rate=0.1,
    verbose=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    T = num_nodes + 1

    # === Chromosome Encoding ===
    def random_policy():
        eta = torch.zeros((num_agents, T, num_nodes + 1), dtype=torch.float32, device=device)
        for a in range(num_agents):
            for t in range(T):
                j = num_nodes if t == T - 1 else random.randint(0, num_nodes - 1)
                eta[a, t, j] = 1.0
        return eta

    def encode(y, eta):
        return torch.cat([y.flatten(), eta.flatten()])

    def decode(chrom):
        y_flat = chrom[: num_nodes * dim]
        eta_flat = chrom[num_nodes * dim :]
        y = y_flat.view(num_nodes, dim)
        eta = eta_flat.view(num_agents, T, num_nodes + 1)
        return y, eta

    # === Cost Function ===
    def cost_function(chrom):
        y, eta = decode(chrom)
        total = 0.0
        for a in range(num_agents):
            prev = s[a].to(device)
            for t in range(T):
                selected_j = torch.argmax(eta[a, t]).item()
                curr = y[selected_j] if selected_j < num_nodes else e[a].to(device)
                total += torch.sum((curr - prev) ** 2)
                prev = curr
        return total.item()

    # === Initial Population ===
    population = []
    for _ in range(pop_size):
        y_init = torch.tensor(Y_init[0], device=device) if Y_init is not None else torch.rand((num_nodes, dim), device=device) * (YMAX - YMIN) + YMIN
        eta_init = random_policy()
        chrom = encode(y_init, eta_init)
        population.append(chrom)

    # === GA Main Loop ===
    for gen in range(generations):
        fitness = torch.tensor([cost_function(ind) for ind in population], device=device)
        sorted_idx = torch.argsort(fitness)
        population = [population[i] for i in sorted_idx]
        elites = population[:5]

        new_population = elites.copy()
        while len(new_population) < pop_size:
            p1, p2 = random.sample(elites, 2)
            mask = torch.rand(p1.shape, device=device) < 0.5
            child = torch.where(mask, p1, p2).clone()

            for i in range(num_nodes * dim):
                if random.random() < mutation_rate:
                    child[i] = torch.rand(1, device=device) * (YMAX - YMIN) + YMIN

            eta_start = num_nodes * dim
            for a in range(num_agents):
                for t in range(T):
                    if random.random() < mutation_rate:
                        base = eta_start + (a * T + t) * (num_nodes + 1)
                        new_j = num_nodes if t == T - 1 else random.randint(0, num_nodes - 1)
                        child[base : base + num_nodes + 1] = 0.0
                        child[base + new_j] = 1.0
            new_population.append(child)

        population = new_population

        if verbose and (gen % 50 == 0 or gen == generations - 1):
            print(f"Generation {gen+1}/{generations}, Best Cost: {fitness[sorted_idx[0]]:.4f}")

    best_y, best_eta = decode(population[0])
    best_cost = cost_function(population[0])
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Finished in {elapsed_time:.2f}s. Best Cost: {best_cost:.4f}")
    return best_y.cpu().numpy(), best_eta.cpu().numpy(), best_cost, elapsed_time

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
        best_y, best_eta, best_cost, elapsed_time = ga_gpu(
            s,
            e,
            num_nodes,
            num_agents,
            dim,
            Y_init=None,
            verbose=True,
            pop_size=100,
            generations=100,
            mutation_rate=0.3,
        )
    print(f"Best Cost: {best_cost:.4f}, Elapsed Time: {elapsed_time:.2f}s")

