# Learning to Solve Many-to-Many Pickup and Delivery Problems with Multi-head Heterogeneous Attention

Implementation for the paper **“Learning to Solve Many-to-Many Pickup and Delivery Problems with Multi-head Heterogeneous Attention”**.


Requirements
------------

- Python 3.10
- Training requires *NVIDIA GPU + CUDA* (uses NCCL + `torchrun` / DDP)

Installation
------------

```bash
conda create -n hap python=3.10 -y
conda activate hap

pip install -r requirements.txt
```
Usage
------------

#### Generate validation data

```bash
python generate_data.py --name validation --graph_sizes 20 --seed 12345
```

#### Generate testing data

```bash
python gen_test_data.py
```

#### Train (single GPU)

```bash
graph_node=20
validation_data_seed=12345

torchrun --nproc_per_node=1 run.py \
  --graph_size $graph_node \
  --run_name "n${graph_node}_main" \
  --val_dataset "data/pdp/pdp${graph_node}_validation_seed${validation_data_seed}.pkl"
```

#### Train (multi-GPU, example 4 GPUs)

```bash
torchrun --nproc_per_node=4 run.py \
  --graph_size 20 \
  --run_name "n20_main" \
  --val_dataset "data/pdp/pdp20_validation_seed12345.pkl"
```

##### Evaluate 

1) Edit the config of `eval_main.py`:
- Set checkpoint paths in `models = {...}`
- Set dataset directory in `data_files = ...`
- Set output directory in `out_dir = ...`

2) Run:

```bash
python -u eval_main.py
```



### Acknowledgement

Thanks to [Wouter Kool et al.](https://github.com/wouterkool/attention-learn-to-route) and [Jingwen Li et al.](https://github.com/jingwenli0312/Heterogeneous-Attentions-PDP-DRL/tree/main) for getting me started with the code.

