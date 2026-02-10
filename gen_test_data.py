import numpy as np

import os
import pickle
import torch
from scipy import stats


def normalize(file_path):

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    

    locations = np.array([data[0][0]] + data[0][1])

    demands = np.array(data[0][2])
    capacity = data[0][3]

    x_min, y_min = locations.min(axis=0)
    x_max, y_max = locations.max(axis=0)

    normalized_locations = (locations - [x_min, y_min]) / ([x_max - x_min, y_max - y_min])

    scaled_demands = demands / capacity

    normalized_capacity = 1

    normalized_locations = normalized_locations.tolist()
    scaled_demands = scaled_demands.tolist()

    out = list(zip([normalized_locations[0]], [normalized_locations[1:]], [scaled_demands], [normalized_capacity]))
    return out


def get_static_dynamic(file_path):

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    all_ = []
    for i in range(len(data)):
        coordinates = [data[i][0]] + data[i][1]


        demand = data[i][2]

        load = -min(-sum(demand), 0)
        demand.insert(0, 0)


        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        tensor_coords = torch.tensor([x_coords, y_coords], dtype=torch.float64)

        
        static = tensor_coords.unsqueeze(0)
        
        demands = torch.tensor(demand, dtype=torch.float32).unsqueeze(0)

        loads = torch.zeros(demands.shape, dtype=torch.float32)

        loads[ :] = load

        dynamic = torch.tensor(np.concatenate((loads, demands), axis=0)).unsqueeze(0)

        all_.append((static, dynamic))

    return all_


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def generate_pdp_data(dataset_size, pdp_size, depot_mode="random"):

    CAPACITIES = {
        10: 40.,
        20: 50.,
        40: 60,
        50: 60.,
        60: 100,
        100: 100.,
        130: 100.,
        150: 100.,
        200 : 100.
    }

    max_demand = 50
    demand_mean = 25  # Mean for normal distribution (adjust as necessary)
    demand_stddev = 10  # Standard deviation for normal distribution (adjust as necessary)
    
    num_clusters = 3  # Number of clusters
    cluster_spread = 0.1  # Spread of nodes around cluster centers

    if depot_mode == "centered":
        # Generate depots near the center (0.5, 0.5)
        depot_locations = (0.5 + np.random.normal(scale=0.05, size=(dataset_size, 2))).tolist()

    elif depot_mode == "corner":
        # Four corners
        corners = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        # Randomly pick a corner for each depot
        chosen_corners = corners[np.random.choice(4, dataset_size)]
        depot_locations = (chosen_corners + np.random.normal(scale=0.05, size=(dataset_size, 2))).tolist()

    elif depot_mode == "random":
        depot_locations = np.random.uniform(size=(dataset_size, 2)).tolist()
    
    node_locations = []
    for i in range(dataset_size):
        cluster_centers = np.random.uniform(size=(num_clusters, 2))
        cluster_node_locations = []
        
        for _ in range(pdp_size):
            cluster_idx = np.random.choice(num_clusters)
            cluster_center = cluster_centers[cluster_idx]
            node_location = cluster_center + np.random.normal(scale=cluster_spread, size=2)
            cluster_node_locations.append(node_location.tolist())  # Convert array to list
        
        node_locations.append(cluster_node_locations)
    
    demands = np.random.normal(loc=demand_mean, scale=demand_stddev, size=(dataset_size, pdp_size)).astype(int)
    
    demands = np.clip(demands, 1, max_demand)

    flat_values = demands.flatten()

    percentage_to_modify = np.random.uniform(0.45, 0.6)
    num_entries_to_modify = int(flat_values.size * percentage_to_modify)

    indices_to_modify = np.random.choice(flat_values.size, num_entries_to_modify, replace=False)

    flat_values[indices_to_modify] *= -1

    demands = flat_values.reshape(dataset_size, pdp_size)
    
    demands = demands.tolist()
    
    capacities = np.full(dataset_size, CAPACITIES[pdp_size]).tolist()

    for i in range(dataset_size):
        paired = list(zip(demands[i], node_locations[i]))
        
        paired.sort(key=lambda x: x[0])
        
        sorted_demands, sorted_locations = zip(*paired)
        
        demands[i] = list(sorted_demands)
        
        node_locations[i] = [list(location) for location in sorted_locations]  

    return list(zip(depot_locations, node_locations, demands, capacities))



def generate_pdp_data_uniform(dataset_size, pdp_size):
    CAPACITIES = {
        10: 40., 20: 50., 40: 60., 50: 60.,
        60: 100., 100: 100., 130: 100.,
        150: 100., 200: 100.
    }

    max_demand = 50
    demand_mean = 25
    demand_stddev = 10

    depot_locations = np.random.uniform(size=(dataset_size, 2)).tolist()
    node_locations = np.random.uniform(size=(dataset_size, pdp_size, 2)).tolist()

    # --- Demands ---
    demands = np.random.normal(loc=demand_mean, scale=demand_stddev, size=(dataset_size, pdp_size)).astype(int)
    demands = np.clip(demands, 1, max_demand)
    flat_values = demands.flatten()

    # Flip a random percentage of demands to negative (pickups)
    percentage_to_modify = np.random.uniform(0.45, 0.6)
    num_entries_to_modify = int(flat_values.size * percentage_to_modify)
    indices_to_modify = np.random.choice(flat_values.size, num_entries_to_modify, replace=False)
    flat_values[indices_to_modify] *= -1
    demands = flat_values.reshape(dataset_size, pdp_size).tolist()

    capacities = np.full(dataset_size, CAPACITIES[pdp_size]).tolist()

    # Sort demands by value to ensure pickups (negative) appear before deliveries (positive)
    for i in range(dataset_size):
        paired = list(zip(demands[i], node_locations[i]))
        paired.sort(key=lambda x: x[0])
        sorted_demands, sorted_locations = zip(*paired)
        demands[i] = list(sorted_demands)
        node_locations[i] = [list(loc) for loc in sorted_locations]

    return list(zip(depot_locations, node_locations, demands, capacities))



def generate_pdp_data_mixed(dataset_size, pdp_size):
    CAPACITIES = {
        10: 1., 20: 1., 40: 1., 50: 1.,
        60: 1., 100: 1., 130: 1.,
        150: 1., 200: 1.
    }

    max_demand = 50
    demand_mean = 25
    demand_stddev = 10
    num_clusters = 3
    cluster_spread = 0.1

    depot_locations = np.random.uniform(size=(dataset_size, 2)).tolist()
    node_locations = []
    demands = []

    for i in range(dataset_size):
        is_clustered = np.random.rand() < 0.5  # 50% chance for cluster / uniform

        if is_clustered:
            # --- Clustered nodes ---
            cluster_centers = np.random.uniform(size=(num_clusters, 2))
            cluster_node_locations = []
            for _ in range(pdp_size):
                cluster_idx = np.random.choice(num_clusters)
                cluster_center = cluster_centers[cluster_idx]
                node_location = cluster_center + np.random.normal(scale=cluster_spread, size=2)
                cluster_node_locations.append(node_location.tolist())
        else:
            # --- Uniform nodes ---
            cluster_node_locations = np.random.uniform(size=(pdp_size, 2)).tolist()

        node_locations.append(cluster_node_locations)

        # --- Demands ---
        d = np.random.normal(loc=demand_mean, scale=demand_stddev, size=pdp_size).astype(int)
        d = np.clip(d, 1, max_demand)

        # Randomly flip 45–60% of them to negative
        percentage_to_modify = np.random.uniform(0.45, 0.6)
        num_to_modify = int(pdp_size * percentage_to_modify)
        indices = np.random.choice(pdp_size, num_to_modify, replace=False)
        d[indices] *= -1
        demands.append(d.tolist())

    capacities = np.full(dataset_size, CAPACITIES[pdp_size]).tolist()

    # --- Sort by demand sign (pickup first) ---
    for i in range(dataset_size):
        paired = list(zip(demands[i], node_locations[i]))
        paired.sort(key=lambda x: x[0])
        sorted_demands, sorted_locations = zip(*paired)
        demands[i] = list(sorted_demands)
        node_locations[i] = [list(loc) for loc in sorted_locations]

    return list(zip(depot_locations, node_locations, demands, capacities))



def generate_pdp_data_gaussian(dataset_size, pdp_size, sigma=0.15):
    CAPACITIES = {
        10: 40., 20: 50., 40: 60., 50: 60.,
        60: 100., 100: 100., 130: 100.,
        150: 100., 200: 100.
    }

    max_demand = 50
    demand_mean = 25
    demand_stddev = 10

    def truncated_normal(size):
        mu, lower, upper = 0.5, 0, 1
        X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        return X.rvs(size)


    depot_locations = np.stack([
        truncated_normal(dataset_size),
        truncated_normal(dataset_size)
    ], axis=1).tolist()

    node_locations = np.stack([
        truncated_normal(dataset_size * pdp_size),
        truncated_normal(dataset_size * pdp_size)
    ], axis=1).reshape(dataset_size, pdp_size, 2).tolist()

    demands = np.random.normal(loc=demand_mean, scale=demand_stddev, size=(dataset_size, pdp_size)).astype(int)
    
    demands = np.clip(demands, 1, max_demand)

    flat_values = demands.flatten()

    percentage_to_modify = np.random.uniform(0.45, 0.6)
    num_entries_to_modify = int(flat_values.size * percentage_to_modify)

    indices_to_modify = np.random.choice(flat_values.size, num_entries_to_modify, replace=False)

    flat_values[indices_to_modify] *= -1

    demands = flat_values.reshape(dataset_size, pdp_size)
    
    demands = demands.tolist()
    
    capacities = np.full(dataset_size, CAPACITIES[pdp_size]).tolist()

    for i in range(dataset_size):
        paired = list(zip(demands[i], node_locations[i]))
        
        paired.sort(key=lambda x: x[0])
        
        sorted_demands, sorted_locations = zip(*paired)
        
        demands[i] = list(sorted_demands)
        
        node_locations[i] = [list(location) for location in sorted_locations]  # Convert tuples back to lists

    return list(zip(depot_locations, node_locations, demands, capacities))





def main(num_nodes, num_ins, mode, file_dir, depot_mode="random", sigma = 1.0):
    np.random.seed(123)

    if mode == "clustered":
        # raise NotImplementedError("Already generated clustered data. Please choose 'uniform' or 'mixed' mode.")

        file_dir = file_dir + f'/cluster/depot_{depot_mode}/' + str(num_nodes) 
        for i in range(num_ins):
            save_path = file_dir + '/' + str(num_nodes) +  '_' + str(i) + '.pkl'
            dataset = generate_pdp_data(1, num_nodes, depot_mode=depot_mode)

            save_dataset(dataset, save_path)

            out_path = os.path.splitext(save_path)[:1][0]+'_normalized'

            save_dataset(normalize(save_path), out_path)

        print(f"Saved clustered instance file: {save_path}")
        
    elif mode == "uniform":
        file_dir = file_dir + '/uniform/' + str(num_nodes) 
        for i in range(num_ins):
            save_path = file_dir + '/' + str(num_nodes) +  '_' + str(i) + '.pkl'
            dataset = generate_pdp_data_uniform(1, num_nodes)

            save_dataset(dataset, save_path)

            out_path = os.path.splitext(save_path)[:1][0]+'_normalized'

            save_dataset(normalize(save_path), out_path)

    elif mode == "mixed":
        file_dir = file_dir + '/mixed/' + str(num_nodes) 
        for i in range(num_ins):
            save_path = file_dir + '/' + str(num_nodes) +  '_' + str(i) + '.pkl'
            dataset = generate_pdp_data_mixed(1, num_nodes)

            save_dataset(dataset, save_path)

            out_path = os.path.splitext(save_path)[:1][0]+'_normalized'

            save_dataset(normalize(save_path), out_path)
        
    elif mode == "gaussian":
        file_dir = file_dir + f'/gaussian_sigma{sigma}/' + str(num_nodes) 
        for i in range(num_ins):
            save_path = file_dir + '/' + str(num_nodes) +  '_' + str(i) + '.pkl'
            dataset = generate_pdp_data_gaussian(1, num_nodes, sigma)

            save_dataset(dataset, save_path)

            out_path = os.path.splitext(save_path)[:1][0]+'_normalized'

            save_dataset(normalize(save_path), out_path)
    else:
        raise ValueError("Invalid mode! Choose from 'clustered', 'uniform',  'mixed', 'gaussian'.")



if __name__ == "__main__":
    num_ins = 100
    file_dir = 'testing_data' 

    
    nodes = [20]
    modes = [ "uniform"]        #clustered

    for num_nodes in nodes:
        for mode in modes:
            main(num_nodes, num_ins, mode, file_dir, depot_mode='centered', sigma=0.8)

