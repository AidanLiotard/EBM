# Parallel Training Trajectory

# Install
The package is still under development, so install as editable:
```bash
pip install -e .
```

# Train a RBM using PTT
To train a RBM using PTT as sampling scheme: 
```bash
ptt train -d /path/to/dataset --num_hiddens <int> --gibbs_steps <int> --num_chains <int> --num_updates <int> --filename /path/to/archive --device <cuda|cpu> --learning_rate <float>
```
The dataset should comply with what is specified in the [rbms](https://dsysdml.github.io/rbms/) repo.

Since PTT is high-level, it works for both Bernoulli-Bernoulli RBM and Potts-Bernoulli RBM and should work for any RBM satisfying the implementation of the `RBM` class from the `rbms` package.