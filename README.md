# meta
A library for multi-task learning and meta-learning.

### SLAW: Scaled Loss Approximate Weighting for Efficient Multi-Task Learning

Code for the paper [SLAW: Scaled Loss Approximate Weighting for Efficient Multi-Task Learning](https://arxiv.org/abs/2109.08218). This branch contains the implementation of SLAW, all the baselines compared against in the paper, and the code needed to perform multi-task training on MTRegression, NYUv2, and PCBA.

Running the experiments from the paper requires that you have all dependencies listed in `requirements.txt` installed. The deepchem package can be annoying to install, because it depends on RDKit. I recommend creating a conda environment and installing rdkit directly through conda, then all other dependencies from `requirements.txt`.

To run the training experiments from the paper, simply run `run_experiments.sh`. The results should populate in the `results` directory. Additional plots can be created with the scripts named `scripts/make_*_plots.py`, though you might have to tinker with the scripts to get them to run properly. The empirical validation of SLAW from the appendix can be run with `scripts/claw_test.py`.

If you run into trouble, feel free to reach out: [mcrawsha@gmu.edu](mcrawsha@gmu.edu).

If you use this code, please cite our paper with the following bibtex entry:
```
@article{crawshaw2021slaw,
  title={SLAW: Scaled Loss Approximate Weighting for Efficient Multi-Task Learning},
  author={Crawshaw, Michael and Ko{\v{s}}eck{\'a}, Jana},
  journal={arXiv preprint arXiv:2109.08218},
  year={2021}
}
```
