import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
import argparse


def generate_pdp_data(dataset_size, pdp_size):

    CAPACITIES = {
        10: 20.,
        20: 30.,
        40: 40,
        50: 40.,
        60: 50,
        100: 50.,
        130: 50.,
        150:50.,
        200: 50.
    }

    max_demand = 9
    
    # Generate random depot and node locations
    depot_locations = np.random.uniform(size=(dataset_size, 2)).tolist()
    node_locations = np.random.uniform(size=(dataset_size, pdp_size, 2)).tolist()
    
    demands = np.random.randint(1, max_demand+1, (dataset_size, pdp_size))

    flat_values = demands.flatten()

    # Calculate the percentage of entries to modify (between 45% and 60%)
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
        node_locations[i] = list(sorted_locations)

    
    
    return list(zip(depot_locations, node_locations, demands, capacities))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='pdp')
    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20],
                        help="Sizes of problem instances (default 20, 40, 80, 120)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=12345, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or len(opts.graph_sizes) == 1, \
        "Can only specify --filename when generating a single graph size"


    for graph_size in opts.graph_sizes:

        datadir = os.path.join(opts.data_dir, opts.problem)
        os.makedirs(datadir, exist_ok=True)

        if opts.filename is None:
            # Example: data/pdp/pdp20_validation_seed1234.pkl
            filename = os.path.join(
                datadir,
                "{}{}_{}_seed{}.pkl".format(opts.problem, graph_size, opts.name, opts.seed)
            )
        else:
            filename = check_extension(opts.filename)

        assert opts.f or not os.path.isfile(check_extension(filename)), \
            "File already exists! Try running with -f option to overwrite."

        np.random.seed(opts.seed)
        if opts.problem == 'pdp':
            dataset = generate_pdp_data(opts.dataset_size, graph_size)
        else:
            assert False, "Unknown problem: {}".format(opts.problem)
        # print(dataset[0])
        save_dataset(dataset, filename)



'''
python generate_data.py --name validation --graph_sizes 20 --seed 12345


'''
