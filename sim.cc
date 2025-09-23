#include <omp.h>

#include <algorithm>
#include <bitset>  // For efficient collision cache
#include <cmath>
#include <fstream>
#include <vector>

#include "collision.h"
#include "io.h"
#include "sim_validator.h"

//* FASTEST VERSION */

// global bitset, avoid storing on the stack
constexpr int MAX_PARTICLES = 1000;  // Adjust based on max expected particles
std::bitset<MAX_PARTICLES * MAX_PARTICLES> collision_matrix;

// Function to get a unique index for cache
inline int get_collision_index(int i, int j) {
    return (i * MAX_PARTICLES) + j;  // Flattened 2D index
}

int main(int argc, char* argv[]) {
    // Read arguments and input file
    Params params{};
    std::vector<Particle> particles;
    read_args(argc, argv, params, particles);
    // Set number of threads
    omp_set_num_threads(params.param_threads);

#if CHECK == 1
    // Initialize collision checker
    SimulationValidator validator(params.param_particles, params.square_size, params.param_radius, params.param_steps);
    validator.initialize(particles);
#endif

    /* SWEEP & PRUNE & GRID */

    // Define cell size & number of cells in the grid
    const int GRID_CELL_SIZE = std::max(
        3 * params.param_radius, (int)std::sqrt((params.square_size * params.square_size / params.param_particles)));
    const int GRID_UNITS_X = params.square_size / GRID_CELL_SIZE;
    const int GRID_UNITS_Y = params.square_size / GRID_CELL_SIZE;
    std::vector<std::vector<std::vector<Particle*>>> grid(GRID_UNITS_X,
                                                          std::vector<std::vector<Particle*>>(GRID_UNITS_Y));

    // Used when looking at surrounding grid cells
    constexpr int neighbor_offsets[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    for (int step = 0; step < params.param_steps; ++step) {
        std::sort(particles.begin(), particles.end(),
                  [](const Particle& a, const Particle& b) { return a.loc.x < b.loc.x; });

        // Update positions
#pragma omp parallel for simd
        for (size_t i = 0; i < particles.size(); i++) {
            particles[i].loc.x += particles[i].vel.x;
            particles[i].loc.y += particles[i].vel.y;
        }

        // Reset the cache per step
        // TODO: Opt
        collision_matrix.reset();

        // Clear grid
#pragma omp parallel for collapse(2)
        for (int y = 0; y < GRID_UNITS_Y; y++) {
            for (int x = 0; x < GRID_UNITS_X; x++) {
                grid[x][y].clear();
            }
        }

        // assign to grid
#pragma omp parallel for
        for (size_t i = 0; i < particles.size(); i++) {
            int grid_x = std::clamp((int)(particles[i].loc.x / GRID_CELL_SIZE), 0, GRID_UNITS_X - 1);
            int grid_y = std::clamp((int)(particles[i].loc.y / GRID_CELL_SIZE), 0, GRID_UNITS_Y - 1);
#pragma omp critical
            grid[grid_x][grid_y].push_back(&particles[i]);
        }

        // process collisions
        bool collisionOccurred;
        do {
            collisionOccurred = false;

            // Wall
#pragma omp parallel for reduction(|| : collisionOccurred)
            for (size_t i = 0; i < particles.size(); i++) {
                if (is_wall_collision(particles[i].loc, particles[i].vel, params.square_size, params.param_radius)) {
                    resolve_wall_collision(particles[i].loc, particles[i].vel, params.square_size, params.param_radius);
                    collisionOccurred = true;
                }
            }

            // Particle (check cache, if not resolve)
#pragma omp parallel for schedule(dynamic)
            for (int x = 0; x < GRID_UNITS_X; x++) {
                for (int y = 0; y < GRID_UNITS_Y; y++) {
                    auto& cell = grid[x][y];

                    if (cell.empty()) continue;

                    // Same cell
                    for (size_t i = 0; i < cell.size(); i++) {
                        for (size_t j = i + 1; j < cell.size(); j++) {
                            int idx = get_collision_index(cell[i] - &particles[0], cell[j] - &particles[0]);

                            if (!collision_matrix[idx]) {  // If not checked before
                                if (is_particle_collision(cell[i]->loc, cell[i]->vel, cell[j]->loc, cell[j]->vel,
                                                          params.param_radius)) {
#pragma omp critical
                                    {
                                        resolve_particle_collision(cell[i]->loc, cell[i]->vel, cell[j]->loc,
                                                                   cell[j]->vel);
                                        collisionOccurred = true;
                                        collision_matrix[idx] = true;  // Store in cache
                                    }
                                }
                            }
                        }
                    }

                    // *Neigbbour, 8 diff directions
                    for (int n = 0; n < 8; n++) {
                        int neighbor_x = x + neighbor_offsets[n][0];
                        int neighbor_y = y + neighbor_offsets[n][1];

                        if (neighbor_x >= 0 && neighbor_x < GRID_UNITS_X && neighbor_y >= 0 &&
                            neighbor_y < GRID_UNITS_Y) {
                            auto& neighbor_cell = grid[neighbor_x][neighbor_y];

                            if (neighbor_cell.empty()) continue;

                            for (Particle* p1 : cell) {
                                for (Particle* p2 : neighbor_cell) {
                                    if (p1 == p2) continue;

                                    int idx = get_collision_index(p1 - &particles[0], p2 - &particles[0]);

                                    if (!collision_matrix[idx]) {  // If not checked before
                                        if (is_particle_collision(p1->loc, p1->vel, p2->loc, p2->vel,
                                                                  params.param_radius)) {
#pragma omp critical
                                            {
                                                resolve_particle_collision(p1->loc, p1->vel, p2->loc, p2->vel);
                                                collisionOccurred = true;
                                                collision_matrix[idx] = true;  // Store in cache
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

        } while (collisionOccurred);

/* Validation after each timestep */
#if CHECK == 1
        validator.validate_step(particles);
#endif
    }

    return 0;
}