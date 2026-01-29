# Constrained-Meta-Reinforcement-Learning-with-Provable-Test-Time-Safety

## Overview

This codebase is developed based on the framework introduced in  
**Identifiability and Generalizability in Constrained Inverse Reinforcement Learning**  
([arXiv:2306.00629](https://arxiv.org/pdf/2306.00629.pdf)).

The repository provides implementations of gridworld environments, core CMDP solvers, visualization utilities, and several safe reinforcement learning and meta-reinforcement learning algorithms used for empirical evaluation.

---

## Code Structure

### `algs/` and `env/`

These directories define the gridworld environment and core algorithmic components.

- The gridworld environment, including the construction of transition dynamics, follows the standard formulation in  
  **Sutton, R. S. and Barto, A. G. _Reinforcement Learning: An Introduction_. MIT Press, 2018.**
- We provide basic utility functions for computing:
  - reward and constraint value functions
  - Q-functions
  - advantage functions
- A linear programming (LP) solver is implemented for computing optimal policies in constrained Markov decision processes (CMDPs).

---

### `visualization/`

This directory contains functions for visualizing policies in the gridworld environment, including state–action preferences and policy structure.

---

### `example/`

This directory contains implementations of the following algorithms:

1. **DOPE+**  
   Yu, K., Lee, D., Overman, W., and Lee, D.  
   *Improved regret bound for safe reinforcement learning via tighter cost pessimism and reward optimism.*  
   Reinforcement Learning Journal, 6:493–546, 2025.

2. **DOPE**  
   Bura, A., HasanzadeZonuzy, A., Kalathil, D., Shakkottai, S., and Chamberland, J.-F.  
   *DOPE: Doubly optimistic and pessimistic exploration for safe reinforcement learning.*  
   NeurIPS, 2022.

3. **Log-Barrier**  
   Ni, T. and Kamgarpour, M.  
   *A safe exploration approach to constrained Markov decision processes.*  
   arXiv:2312.00561, 2023.

4. **Safe Meta-RL**  
   Xu, S. and Zhu, M.  
   *Efficient safe meta-reinforcement learning: Provable near-optimality and anytime safety.*  
   NeurIPS, 2025.

5. **Safe-PCE (Ours)**  
   Proposed in *Constrained Meta Reinforcement Learning with Provable Test-Time Safety*.

   In addition, we include a baseline that adapts:
   - Ye, H., Chen, X., Wang, L., and Du, S. S.  
     *On the power of pre-training for generalization in RL: Provable benefits and hardness.*  
     ICML, 2023

   to the constrained meta-RL setting **without** safe exploration guarantees.

---

## Experimental Data

All experimental results, including:
- test-time reward regret
- constraint values at each iteration

for all implemented algorithms are saved in:
