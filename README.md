# collision-simulator
particle collision simulator, parallelised with OpenMP

### Notes
This project was run on a slurm cluster, so it is difficult to replicate on a personal laptop. However, the core code is still provided!

Particles undergo elastic collisions over timesteps.
Since this is a computationally intensive, a few optimisations are implemented. <br>
- The space is split into grids
- Sweep & Prune is applied
- Caching is applied
- And of course, parallelisation with OpenMP
