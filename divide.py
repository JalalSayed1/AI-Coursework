import numpy as np

def divide_into_even_subsets(array, subset_percentage=10):
    """
    Divides an array into smaller subsets, each approximately of a specified percentage size of the original array.
    The function evenly distributes any extra elements among the subsets.
    
    :param array: The original NumPy array to divide.
    :param subset_percentage: The size of each subset as a percentage of the original array.
    :return: A list of NumPy arrays, each being a subset of the original array.
    """
    total_size = len(array)
    subset_base_size = total_size * subset_percentage // 100
    extra_elements = total_size - (subset_base_size * 10)
    
    subsets = []
    start_index = 0

    for i in range(10):
        end_index = start_index + subset_base_size + (1 if i < extra_elements else 0)
        subsets.append(array[start_index:end_index])
        start_index = end_index

    return subsets

# Example usage:
# Assuming you have a NumPy array named 'your_array', you can use the function like this:
subsets = divide_into_even_subsets([i for i in range(51)])
for subset in subsets:
    print(subset, len(subset))


def find_most_representative_subset(subsets, full_data):
    """
    Finds the subset that is most representative of the full dataset based on mean and standard deviation.
    
    :param subsets: A list of NumPy arrays (subsets).
    :param full_data: The original full dataset as a NumPy array.
    :return: The index of the most representative subset.
    """
    full_mean = np.mean(full_data)
    full_std = np.std(full_data)

    closest_mean_index = None
    closest_mean_diff = float('inf')

    for i, subset in enumerate(subsets):
        subset_mean = np.mean(subset)
        subset_std = np.std(subset)

        mean_diff = abs(subset_mean - full_mean)
        std_diff = abs(subset_std - full_std)

        # Combine the differences in mean and standard deviation
        total_diff = mean_diff + std_diff

        if total_diff < closest_mean_diff:
            closest_mean_diff = total_diff
            closest_mean_index = i

    return closest_mean_index

# Example usage:
# Assuming you have a list of subsets named 'subsets' and the original array 'your_array', you can use the function like this:
# most_representative_subset_index = find_most_representative_subset(subsets, your_array)
# print("Most representative subset index:", most_representative_subset_index)
index = find_most_representative_subset(subsets, np.asarray([i for i in range(51)]))
print(subsets[index])
