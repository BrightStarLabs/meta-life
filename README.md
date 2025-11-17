# meta-life
Food foraging and poison avoidance agent training.
Agent must find food and avoid poison.


dependencies:
- Pygame
- default scientific (numpy, matplotlib, etc..)
- Taichi (only for the adaptation one)


### Scripts

-> meta-life-adr
uses an EM based on a kind of Advection Reaction Diffusion equations, discretised in cells
local update only + nonlinearity

-> meta-life-reservoir
uses differential equation based reservoir (also called liquid state) Neural network - based EM.
fully connected reservoir + nonlinearity


-> meta-life-adaptation
uses reactive-SWEM (reaction term per cell + SWEM)
SWEM: small-world EM: convolution kernel + small world connections + nonlinearity
adaptation mode: food flips into poison every random timesteps, this elicits the emergence of online-adaptation policies, a primitive form of meta-learning
