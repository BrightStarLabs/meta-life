# meta-life
Food foraging and poison avoidance agent training.
Agent must find food and avoid poison.


dependencies (in the requirements.txt):
- Pygame
- default scientific (numpy, matplotlib, etc..)
- Taichi (only for the adaptation one)

## Scripts

### meta-life-adr
This model uses an EM based on a kind of Advection Reaction Diffusion equations, discretised in cells.
**local update only + nonlinearity**

### meta-life-reservoir
This model uses differential equation based reservoir (also called liquid state) Neural network - based EM.
**fully connected reservoir + nonlinearity**


### meta-life-adaptation
This model uses reactive-SWEM (reaction term per cell + SWEM) where
SWEM (small-world EM) is based on **convolution kernel + small world connections + nonlinearity**.
In **adaptation mode**: food flips into poison every random timesteps, this elicits the emergence of **online-adaptation policies, a primitive form of meta-learning**
