# Swarm Environment

Simple PyBullet Environments for a swarm of mobile robots. Developed for the [Collab](https://collab.me.vt.edu/).

## Quickstart

Use these commands to quickly get set up with this repository, and start up the racecars.

```bash
# Clone the Repository
git clone https://github.com/VT-Collab/swarm-pybullet.git
cd swarm-pybullet

# Create a Conda Environment using `environment.yml`
conda env create -f environment.yml
```
If you get an error related to `libcxx`, include `conda-forge` to your conda channels by typing using the commands below.
```bash
# Add conda-forge to conda channels
conda config --add channels conda-forge
conda config --set channel_priority strict
```
## Start-Up (from Scratch)

Use these commands if the above quick-start instructions don't work for you, or you prefer to maintain your own Python
packaging solution, and want to make sure the dependencies are in check. If you're just trying to use the code, look at
the Quickstart section above.

```bash
# Create & Activate Conda Environment
conda create --name panda-env python=3.7
conda activate swarm-pybullet

# Install Dependencies
pip install pybullet numpy
```

## How to Run

This repo includes one sample environment, as well as the utility functions for assests (i.e., objects to interact with).

To see this sample environments, run:
```bash
python3 main.py
```
