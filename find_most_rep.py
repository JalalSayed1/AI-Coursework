import numpy as np
import matplotlib.pyplot as plt

def find_most_representative_subset_with_plot(subsets, full_data):
    """
    Finds the subset that is most representative of the full dataset based on mean and standard deviation,
    and plots the subsets with the most representative one highlighted.
    
    :param subsets: A list of NumPy arrays (subsets).
    :param full_data: The original full dataset as a NumPy array.
    :return: The index of the most representative subset.
    """
    full_mean = np.mean(full_data)
    full_std = np.std(full_data)

    closest_mean_index = None
    closest_mean_diff = float('inf')

    # Prepare data for plotting
    subset_means = []
    subset_stds = []

    for i, subset in enumerate(subsets):
        subset_mean = np.mean(subset)
        subset_std = np.std(subset)
        subset_means.append(subset_mean)
        subset_stds.append(subset_std)

        mean_diff = abs(subset_mean - full_mean)
        std_diff = abs(subset_std - full_std)

        total_diff = mean_diff + std_diff

        if total_diff < closest_mean_diff:
            closest_mean_diff = total_diff
            closest_mean_index = i

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot each subset
    for i, subset in enumerate(subsets):
        plt.scatter(np.full(subset.shape, i), subset, alpha=0.5)

    # Highlight the most representative subset
    plt.scatter(np.full(subsets[closest_mean_index].shape, closest_mean_index), 
                subsets[closest_mean_index], color='red', label='Most Representative Subset')

    # Plot the mean and std deviation lines
    plt.axhline(full_mean, color='green', linestyle='--', label='Full Dataset Mean')
    plt.axhline(full_mean + full_std, color='green', linestyle=':', label='Full Dataset Std Dev')
    plt.axhline(full_mean - full_std, color='green', linestyle=':')

    plt.title("Subsets of the Dataset and the Most Representative Subset")
    plt.xlabel("Subset Index")
    plt.ylabel("Value")
    plt.legend()
    print(f"Most representative subset index: {closest_mean_index} because it has the closest mean and std deviation to the full dataset.")
    plt.show()

    return closest_mean_index

# Example usage with a sample dataset
np.random.seed(0)  # For reproducibility
full_data = np.random.normal(0, 1, 1000)  # A sample dataset
subsets = np.array_split(full_data, 10)  # Splitting into 10 subsets

# Call the function with plotting
most_representative_subset_index = find_most_representative_subset_with_plot(subsets, full_data)
print("Most representative subset index:", most_representative_subset_index)
