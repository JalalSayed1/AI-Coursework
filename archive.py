def update_accuracy_loss(node, accuracy, loss):
    node["Accuracy"] = accuracy
    node["Loss"] = loss
    
def add_child(node, counter, accuracy, loss, pruned_indexes={}, next_states=[]):
    child_node = {"State": f"X{counter}", "Accuracy" : accuracy, "Loss" : loss, "Pruned" : pruned_indexes, "Next States": next_states}
    node["Next States"].append(child_node)

# def heuristic_to_identify_samples(x_train, y_train, num_samples_to_remove):
#     # let [0, 2, 3, 5, 6, 8, 9] have 60% chance of being mislabeled.
#     # let [1, 4, 7] have 23% chance of being mislabeled.
#     # let all have 17% chance of being correctly labeled.
#     # remove the samples with the highest probability of being mislabeled.
    
#     indexes_to_remove = []
    
#     # indexes_to_remove.extend(np.random.choice(np.where(y_train == 0)[0], size=int(num_samples_to_remove * 0.6), replace=False))

#     return indexes_to_remove
    
def heuristic_to_identify_samples(x_train, y_train, num_samples_to_remove):
    high_risk_labels = [0, 2, 3, 5, 6, 8, 9]
    medium_risk_labels = [1, 4, 7]

    sample_risks = []

    for index, label in enumerate(y_train):
        if label in high_risk_labels:
            risk = 0.6  # 60% chance of being mislabeled
        elif label in medium_risk_labels:
            risk = 0.23  # 23% chance

        sample_risks.append((index, risk))

    # Sort samples by descending risk (higher risk first)
    # sorted_samples = sorted(sample_risks, key=lambda x: x[1], reverse=True)

    # Select top 'num_samples_to_remove' indexes
    # indexes_to_remove = [index for index, risk in sorted_samples[:num_samples_to_remove]]

    num_of_high_risk = int(num_samples_to_remove * 0.6)
    num_of_medium_risk = int(num_samples_to_remove * 0.23)
    
    indexes_to_remove = []
    
    #! risk this loop doesn't terminate bc we might not have enough samples to remove:
    while num_of_high_risk > 0 and len(indexes_to_remove) < num_samples_to_remove:
        index = np.random.choice(np.where(y_train == random.choice(high_risk_labels))[0], size=1, replace=False)
        if not index in indexes_to_remove:
            indexes_to_remove.extend(index)
            num_of_high_risk -= 1
            
    while num_of_medium_risk > 0 and len(indexes_to_remove) < num_samples_to_remove:
        index = np.random.choice(np.where(y_train == random.choice(medium_risk_labels))[0], size=1, replace=False)
        if not index in indexes_to_remove:
            indexes_to_remove.extend(index)
            num_of_medium_risk -= 1

    return indexes_to_remove


def retrain_model(x_train, y_train, total_num_samples, current_counter, remove_percentage=0.001):
    num_samples_to_remove = int(total_num_samples * remove_percentage)

    # Implement heuristic here
    # Example: Identify indexes of samples most likely mislabeled or confusing
    indexes_to_remove = heuristic_to_identify_samples(x_train, y_train, num_samples_to_remove)

    temp_pruned_indexes = {index: index for index in indexes_to_remove}
    
    # Retrain model with pruned dataset
    loss, accuracy = trainAndEvaluateModel(x_train, y_train_onehot, x_test, y_test_onehot, model, temp_pruned_indexes)

    # Update tracking and tree structure
    new_counter = current_counter + 1
    add_child(root, new_counter, accuracy, loss, temp_pruned_indexes, [])
    
    return new_counter, accuracy, temp_pruned_indexes

