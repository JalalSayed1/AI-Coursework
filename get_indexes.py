def get_random_index(y_train, risk_labels):
    # Find indices of all elements in y_train that match any of the risk labels
    risk_indices = np.where(np.isin(y_train, risk_labels))[0]

    # If there are no matching indices, return None or handle as needed
    if len(risk_indices) == 0:
        return None

    # Randomly select an index from the risk indices
    return np.random.choice(risk_indices, size=1, replace=False)[0]

def heuristic_to_identify_samples(x_train, y_train, num_samples_to_remove):
    #! ASSUMPTION: might have to have num_samples_to_remove be less than len(y_train or x_train) for this to work
    high_risk_labels = [0, 2, 3, 5, 6, 8, 9]
    medium_risk_labels = [1, 4, 7]

    # sample_risks = []

    # for index, label in enumerate(y_train):
    #     if label in high_risk_labels:
    #         risk = 0.6  # 60% chance of being mislabeled
    #     elif label in medium_risk_labels:
    #         risk = 0.23  # 23% chance

    #     sample_risks.append((index, risk))

    # Sort samples by descending risk (higher risk first)
    # sorted_samples = sorted(sample_risks, key=lambda x: x[1], reverse=True)

    # Select top 'num_samples_to_remove' indexes
    # indexes_to_remove = [index for index, risk in sorted_samples[:num_samples_to_remove]]

    num_of_high_risk = int(num_samples_to_remove * 0.6)
    num_of_medium_risk = int(num_samples_to_remove * 0.23)
    possible_num_of_index_to_remove = num_of_high_risk + num_of_medium_risk
    
    # a set of indexes to remove:
    indexes_to_remove = set()
    
    
    #! risk this loop doesn't terminate bc we might not have enough samples to remove:
    # num_of_high_risk > 0 and
    # while len(indexes_to_remove) < num_samples_to_remove and (num_of_high_risk > 0 or num_of_medium_risk > 0):
    # while len(indexes_to_remove) < possible_num_of_index_to_remove:
    print(f"len(indexes_to_remove): {len(indexes_to_remove)}")
    # index = np.random.choice(np.where(y_train == random.choice(high_risk_labels))[0], size=1, replace=False)
    # get num_of_high_risk indexes from data:
    while num_of_high_risk > 0 and len(indexes_to_remove) < possible_num_of_index_to_remove:
        #! check if index is None:
        #! ASSUMPTION: if there is no enough elements, get_random_index might return the same index over and over again. preventing the loop from terminating:
        index = get_random_index(y_train, high_risk_labels)
        if not index in indexes_to_remove:
            indexes_to_remove.add(index)
            num_of_high_risk -= 1
            
    print(f"len(indexes_to_remove) 2: {len(indexes_to_remove)}")
    while num_of_medium_risk > 0 and len(indexes_to_remove) < possible_num_of_index_to_remove:
        #! check if index is None:
        index = get_random_index(y_train, medium_risk_labels)
        if not index in indexes_to_remove:
            indexes_to_remove.add(index)
            num_of_medium_risk -= 1
        
        
        # if not index in indexes_to_remove:
        #     indexes_to_remove.extend(index)
        #     num_of_high_risk -= 1
            
    # while num_of_medium_risk > 0 and len(indexes_to_remove) < num_samples_to_remove:
    #     index = np.random.choice(np.where(y_train == random.choice(medium_risk_labels))[0], size=1, replace=False)
    #     if not index in indexes_to_remove:
    #         indexes_to_remove.extend(index)
    #         num_of_medium_risk -= 1

    return indexes_to_remove


import numpy as np
import random

def test_heuristic_to_identify_samples():
    # Create a mock dataset
    y_train_mock = np.array([0, 2, 3, 1, 4, 7, 5, 6, 8, 9, 0, 2])  # Example labels
    num_samples_to_remove = len(y_train_mock)-2  # Example number of samples to remove

    # Expected outcome
    expected_high_risk_labels = [0, 2, 3, 5, 6, 8, 9]
    expected_indexes = []

    for i, label in enumerate(y_train_mock):
        if label in expected_high_risk_labels:
            expected_indexes.append(i)
            if len(expected_indexes) == num_samples_to_remove:
                break

    # Call the function
    actual_indexes = heuristic_to_identify_samples(None, y_train_mock, num_samples_to_remove)
    print(f"{len(actual_indexes)} != {num_samples_to_remove}")
    # Assert the results
    # assert len(actual_indexes) == num_samples_to_remove, "Incorrect number of samples removed"
    # assert all([index in expected_indexes for index in actual_indexes]), "Incorrect indexes identified for removal"

if __name__ == "__main__":
    # Run the test
    test_heuristic_to_identify_samples()
