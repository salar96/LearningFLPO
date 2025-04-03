import numpy as np

def generate_dataset(n, cov, k, n_clusters,  scale, seed):
    # Generate random means for each cluster
    np.random.seed(seed)
    means = np.random.rand(n_clusters, 2)*scale

    # Generate random points in k clusters
    nodes = np.zeros((n, 2))
    for i in range(n):
        cluster = np.random.randint(0, means.shape[0])  # Choose a random cluster
        nodes[i] = np.random.multivariate_normal(means[cluster], cov)
    Y_s = np.tile(np.sum(nodes, axis=0), (k,1))/n + np.random.rand(k, 2)*scale*0.1
    dest = np.array([[0.9,0.8]])*scale
    return nodes,Y_s,dest
