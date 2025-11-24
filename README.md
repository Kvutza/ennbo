# Epistemic Nearest Neighbors
A fast, alternative surrogate for Bayesian optimization

- ENN model, [`EpistemicNearestNeighbors`](https://github.com/yubo-research/enn/blob/main/src/enn/core.py) [1]
- TuRBO-ENN optimizer, class `Turbo` has four modes
	- `TURBO_ONE` - A clone of the TuRBO [2] reference [code](https://github.com/uber-research/TuRBO), reworked to have an `ask()`/`tell()` interface.
	- `TURBO_ENN` - Same as TURBO_ONE, except uses ENN instead of GP and Pareto(mu, se) instead of Thompson sampling.
	- `TURBO_ZERO` - Same as TURBO_ONE, except randomly-chosen RAASP [3] candidates are picked to be proposals. There is no surrogate.
	- `LHD_ONLY` - Just creates an LHD design for every `ask()`. Good for a baseline and testing.

[1] **Sweet, D., & Jadhav, S. A. (2025).** Taking the GP Out of the Loop. *arXiv preprint arXiv:2506.12818*.  
   https://arxiv.org/abs/2506.12818  
[2] **Eriksson, D., Pearce, M., Gardner, J. R., Turner, R., & Poloczek, M. (2020).** Scalable Global Optimization via Local Bayesian Optimization. *Advances in Neural Information Processing Systems, 32*.  
   https://arxiv.org/abs/1910.01739  
[3] **Rashidi, B., Johnstonbaugh, K., & Gao, C. (2024).** Cylindrical Thompson Sampling for High-Dimensional Bayesian Optimization. *Proceedings of The 27th International Conference on Artificial Intelligence and Statistics* (pp. 3502–3510). PMLR.  
   https://proceedings.mlr.press/v238/rashidi24a.html  
