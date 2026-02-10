#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

# from options_deb import get_options
from options import get_options

from train import train_epoch, validate, get_inner_model
from rollout_baselines import NoBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from problems import PDP

def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])  # Local GPU ID on the current node
    torch.cuda.set_device(local_rank)           # Set device for current process
    return local_rank

def run(opts):

    local_rank = setup()

    if dist.get_rank() == 0:
        pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if dist.get_rank() == 0 and not opts.no_tensorboard:
        
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    if dist.get_rank() == 0:
        os.makedirs(opts.save_dir, exist_ok=True)
        # Save arguments so exact configuration can always be found
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device(f"cuda:{local_rank}")

    # Figure out what's the problem
    problem = PDP

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        latent_dim=opts.latent_dim,
    ).to(opts.device)


    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Initialize baseline
    if opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])


    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Skip bias and normalization layers from weight decay
        if name.endswith('.bias') or 'norm' in name.lower() or 'normalizer' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)


    # Initialize optimizer
    optimizer = torch.optim.Adam(
        [
            {'params': decay_params, 'lr': opts.lr_model, 'weight_decay': opts.weight_decay},
            {'params': no_decay_params, 'lr': opts.lr_model, 'weight_decay': 0.0},
        ]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    
    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )

    dist.destroy_process_group()

if __name__ == "__main__":
    run(get_options())

    # nohup python3 run.py --epoch_size 12800 --n_epochs 20 --baseline rollout --val_dataset data/pdp/pdp20_validation_seed1234.pkl  --no_progress_bar > run.log 2>&1

'''

TO RUN:

graph_node=20
validation_data_seed=12345

torchrun --nproc_per_node=1 run.py \
  --graph_size $graph_node \
  --run_name "n${graph_node}_main" \
  --val_dataset "data/pdp/pdp${graph_node}_validation_seed${validation_data_seed}.pkl"     


'''