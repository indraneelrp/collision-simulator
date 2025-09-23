# Collision-simulator
Particle collision simulator, parallelised with OpenMP.<br>
Main logic in **sim.cc**, collision rules in **collision.h**

### Notes
This project and its test cases were run on a slurm cluster, so it is difficult to replicate on a personal laptop. However, the main logic used is in this repo!

Particles undergo elastic collisions over timesteps.
Since this is a computationally intensive, a few optimisations are implemented. <br>
- The space is split into grids
- Sweep & Prune is applied
- Caching is applied
- And of course, parallelisation with OpenMP
