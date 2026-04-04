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
