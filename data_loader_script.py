import torch
from torch.utils.data import Dataset, DataLoader

# Define the dataset
class TSPSyntheticDataset(Dataset):
    def __init__(self, num_samples, num_cities, input_dim):
        self.num_samples = num_samples
        self.num_cities = num_cities
        self.input_dim = input_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return torch.rand(self.num_cities, self.input_dim)

# Function to create the DataLoader
def create_data_loader(batch_size, num_samples, num_cities, input_dim, num_workers=4):
    dataset = TSPSyntheticDataset(num_samples, num_cities, input_dim)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    return data_loader
def create_evaluation_loader(num_samples, num_cities, input_dim, batch_size, num_workers=4):
   
    # Create the evaluation dataset
    eval_dataset = TSPSyntheticDataset(num_samples=num_samples, num_cities=num_cities, input_dim=input_dim)

    # Create the DataLoader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for evaluation
        num_workers=num_workers,
        pin_memory=True,  # Pin memory for faster GPU transfer
    )
    return eval_loader
if __name__ == "__main__":
    # This part runs only if the script is executed directly
    print("Testing DataLoader...")
    batch_size = 128
    num_samples = 1000
    num_cities = 50
    input_dim = 2

    data_loader = create_data_loader(batch_size, num_samples, num_cities, input_dim, num_workers=1)
    for batch in data_loader:
        print(batch.shape)  # Print batch shapes for testing
        break
