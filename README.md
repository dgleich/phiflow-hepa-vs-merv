# Comparing HEPA to MERV filters with a CFD experiment.
This repository contains a simulation comparing the efficiency of HEPA and MERV13 furnace filters in clearing smoke from a 2D room environment. The purpose of this experiment is to illustrate the importance of the clear air delivery rate (CADR) over simple filtration efficiency.

## Simulation Overview

The simulation visualizes smoke dispersion and filtration in a closed room with two different setups: one using a highly restrictive HEPA filter and the other using a 70% efficient MERV13 furnace filter with a higher possible flow. 

The simulation uses the velocity field induced by a set of fans to advect smoke particle concentrations around the simulation. The only physics for particle removal is a filter region contained within a fan. Fans are modeled as boundary conditions with fixed velocity. To filter the particles, we simulate the size of the region that would have gone through the filter at that velocity and remove a fractional concentration (70% or 99.9%) from both that region and the region that would have diffused there as well. 

## How to Run the Simulation

*Prerequisities*
- Python 3.x
- Git
- PhiFlow (we'll show you how to install)

We use [PhiFlow](https://github.com/tum-pbs/PhiFlow) for the CFD routines. 

```bash
git clone https://github.com/dgleich/phiflow-hepa-vs-merv
```

Install phiflow

```bash 
python3 -m venv phiflow-env
source phiflow-env/bin/activate
pip3 install phiflow
```

Then it should run!

```
python3 simulation-code-2024-video-2.py
```

Results will be generated as plots and saved images documenting the filtration process over time. It will also output the concentrations to a CSV file if you wish to analyze them. 

## Notes 
It's possible the smoke values aren't perfectly conserved if you change simulation parameters (or even for these ones). You can usually adjust this by decreasing the timestep. I tolerate a few percent increase or decrease before I get worried. 

