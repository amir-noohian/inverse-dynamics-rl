# Torque-based visual reacher with RL

This repository is for the task of torque-based visual reacher with reinforcement learning. 

## Supported Algorithms
- Soft Actor Critic (SAC)

## Installation instructions
1. Install miniconda or anaconda
2. Create a virtual environment:
```bash
conda create --name myenv python=3.6    # Python 3.6 is necessary
conda activate myenv
```
3. Install packages with:
```bash
pip install -r requirements.txt
pip install .
```

## Run experiment
```python
 python task_ur5_visual_reacher.py  --work_dir "./results" --mode 'l' --seed 0 --env_steps 200100 
```


