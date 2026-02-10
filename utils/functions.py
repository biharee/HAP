import warnings

import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F


def load_problem(name):
    from problems import TSP, PDP, CVRP, SDVRP, OP, PCTSPDet, PCTSPStoch
    problem = {
        'pdp': PDP,
        'tsp': TSP,
        'cvrp': CVRP,
        'sdvrp': SDVRP,
        'op': OP,
        'pctsp_det': PCTSPDet,
        'pctsp_stoch': PCTSPStoch,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, epoch=None):
    from nets.attention_model import AttentionModel

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    problem = load_problem(args['problem'])

    model_class = {
        'attention': AttentionModel
                }.get(args.get('model', 'attention'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        args['embedding_dim'],
        args['hidden_dim'],
        problem,
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None)
    )
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts

def count_demands(demand: torch.Tensor) -> list:
    # Compute the negative and positive counts in one pass

    negative_count = (demand < 0).sum(dim=1)
    positive_count = (demand > 0).sum(dim=1)
    
    # Create a list of tuples for each batch
    return [(neg.item(), pos.item()) for neg, pos in zip(negative_count, positive_count)]


def extract_h_pick_and_delivery(h: torch.Tensor, n_pick, batch_idx: int) -> tuple:
    # Extract the count for the specified batch
    # n_pick, n_delivery = count_demands[batch_idx]
    
    # Extract h_pick and h_delivery for the specified batch
    h_pick = h[batch_idx, 1:n_pick+1, :].unsqueeze(0)         # Extract the first n_pick elements from dim 1
    h_delivery = h[batch_idx, n_pick+1:, :].unsqueeze(0)    # Extract the remaining n_delivery elements from dim 1
    
    return h_pick, h_delivery


def split_features(features):
    batch_size, num_nodes, num_features = features.shape
    
    # Mask for pickup and delivery
    pick_mask = features[:, :, 2] < 0  # Negative values
    del_mask = features[:, :, 2] > 0  # Non-negative values
    
    # Extract values based on mask
    features_pick = torch.zeros_like(features)  # Initialize with zeros
    features_del = torch.zeros_like(features)  # Initialize with zeros
    
    # Fill extracted values
    features_pick[pick_mask] = features[pick_mask]
    features_del[del_mask] = features[del_mask]
    
    return features_pick, features_del


def concatenate_embeddings(embed_pick, embed_delivery, features_pick, features_delivery):

    batch_size, seq_len, embedding_dim = embed_pick.shape

    # Get valid pickup and delivery nodes
    pick_mask = features_pick[:, :, 2] < 0  # Pickup mask
    del_mask = features_delivery[:, :, 2] > 0  # Delivery mask

    # Initialize a zero tensor to store ordered embeddings
    combined_embedd = torch.zeros_like(embed_pick, device=embed_pick.device)  # Same shape: (batch_size, seq_len, embedding_dim)

    # Assign values while maintaining the sequence length
    combined_embedd[pick_mask] = embed_pick[pick_mask]  
    combined_embedd[del_mask] = embed_delivery[del_mask]  
    
    return combined_embedd

def split_embeddings(h, pd_counts):
    """
    Splits the combined embeddings into pickup and delivery tensors.
    Delivery embeddings are right-aligned (padded on the left).
    
    Args:
        h (torch.Tensor): (batch_size, seq_len, embedding_dim), starts with depot.
        pd_counts (List[Tuple[int, int]]): List of (pickup_count, delivery_count) per batch.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: pickup_embeds, delivery_embeds
    """
    batch_size, seq_len, embedding_dim = h.shape
    pd_tensor = torch.tensor(pd_counts, device=h.device)
    pickup_counts = pd_tensor[:, 0]
    delivery_counts = pd_tensor[:, 1]

    max_pickup = pickup_counts.max().item()
    max_delivery = delivery_counts.max().item()

    # Prepare pickup indices
    pickup_range = torch.arange(max_pickup, device=h.device).unsqueeze(0)
    pickup_mask = pickup_range < pickup_counts.unsqueeze(1)
    pickup_indices = 1 + pickup_range.expand(batch_size, -1)
    pickup_indices = torch.where(pickup_mask, pickup_indices, torch.zeros_like(pickup_indices)).long()

    # Batch indexing
    batch_idx = torch.arange(batch_size, device=h.device).unsqueeze(1)

    # Gather pickup embeddings
    pickup_embeds = h[batch_idx, pickup_indices] * pickup_mask.unsqueeze(2)

    # Prepare delivery indices
    delivery_range = torch.arange(max_delivery, device=h.device).unsqueeze(0)
    delivery_mask = delivery_range < delivery_counts.unsqueeze(1)
    delivery_indices = 1 + pickup_counts.unsqueeze(1) + delivery_range.expand(batch_size, -1)
    delivery_indices = torch.where(delivery_mask, delivery_indices, torch.zeros_like(delivery_indices)).long()

    # Gather raw delivery embeddings
    raw_delivery_embeds = h[batch_idx, delivery_indices] * delivery_mask.unsqueeze(2)

    # Right-align the deliveries
    delivery_output = torch.zeros((batch_size, max_delivery, embedding_dim), device=h.device)

    # Compute destination indices for right-aligning
    dest_indices = delivery_range + (max_delivery - delivery_counts).unsqueeze(1)
    dest_indices = torch.where(delivery_mask, dest_indices, torch.zeros_like(delivery_range))

    # Scatter into final tensor
    delivery_output.scatter_(1, dest_indices.unsqueeze(2).expand(-1, -1, embedding_dim), raw_delivery_embeds)

    return pickup_embeds, delivery_output