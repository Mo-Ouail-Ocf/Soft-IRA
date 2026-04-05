# IRA

Reinforcement learning experiments for `SAC`, `IRA`, and `SoftIRA` with Hydra configs, TensorBoard / offline W&B logging, and best/last checkpointing. 

## Install

```bash
conda create -n ira python=3.12 -y
conda activate ira
pip install -r requirements.txt
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Run

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ira
python main.py experiment=b4_softira environment.name=HalfCheetah-v5 run.seed=42
```

SAC baseline:

```bash
python main.py experiment=b1_sac environment.name=HalfCheetah-v5 run.seed=42
```

Note : Code runs only on linux because of use of `faiss-gpu`


Use the `ira` env and call `main.py` with Hydra overrides.

Basic pattern:
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ira
python main.py algorithm=softira environment.name=HalfCheetah-v3 run.seed=42
```

Using an experiment preset:
```bash
python main.py experiment=b4_softira environment.name=HalfCheetah-v3 run.seed=42
```

SAC baseline:
```bash
python main.py experiment=b1_sac environment.name=HalfCheetah-v3 run.seed=42
```

IRA baseline:
```bash
python main.py experiment=b3_ira environment.name=HalfCheetah-v3 run.seed=42
```

Ablation example:
```bash
python main.py experiment=a1_rde_only environment.name=HalfCheetah-v3 run.seed=42
```

Useful overrides:
```bash
python main.py experiment=b4_softira environment.name=Hopper-v3 run.seed=0 run.device=cuda:0
python main.py experiment=o1_softira_double_q environment.name=Ant-v3 run.seed=1 run.device=cuda:0
python main.py algorithm=softira environment.name=HalfCheetah-v3 run.seed=42 logging.wandb=true logging.tensorboard=true
```

Outputs go to:
- run artifacts: `outputs/{algo}_{env}_seed_{seed}_{experiment}`
- Hydra logs: `outputs/hydra/{algo}_{env}_seed_{seed}_{experiment}`

If you want, I can give you the exact commands for all baseline and ablation runs from your markdown.