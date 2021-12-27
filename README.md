# Lagrangian Mechanics Three Ways

This is the source code for my [blog
post](https://adamheins.com/blog/lagrangian-mechanics-three-ways) on
implementing a robot's dynamic equations of motion three ways:
1. with manual differentiation,
2. with automatic differentiation,
3. with symbolic differentiation.

The Jupyter notebook containing the code can be run in the browser
[here](https://mybinder.org/v2/gh/adamheins/lagrangian-mechanics-3-ways/HEAD?filepath=LagrangeThreeWays.ipynb).

Alternatively, you can download the repo and open the notebook yourself. It is
tested with Python 3.7. The repo uses
[pipenv](https://pipenv.pypa.io/en/latest/) for dependency/virtualenv
management. Run:
```
git clone https://github.com/adamheins/lagrangian-mechanics-3-ways
cd lagrangian-mechanics-3-ways
pipenv install
pipenv run jupyter notebook
```
