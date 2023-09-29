#include <iostream>
#include <cmath>
#include <sstream>
#include <vector>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define DEVICE __host__ __device__

#define PI           3.14159265358979323846 

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func <<" "<<cudaGetErrorString(result)<< "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

typedef curandStatePhilox4_32_10_t RNG;

// Radius of particles
const double RADIUS = 1.0;
const int N = 1000000; // Number of particles
const double dt = 0.05; // Time step
const double T = 64.0; // Temperature
const double GAMMA = 1.0; // Drag coefficient
const double mass = 1.0; // Mass of particles
const int STEPS = 10000; // Number of simulation steps

//Sim Box parameters
const int windowWidth = 800;
const int windowHeight = 600;


struct Particle {
    double x = 0;
    double y = 0;
    double vx = 0;
    double vy = 0;

    int pid = 0;

    DEVICE Particle(double x, double y) : x(x), y(y) 
    {

    }

    DEVICE void update(double dx, double dy) {
        x += dx; 
        if(x < 0)
            x = 0;
        else if(x > windowWidth)
            x = windowWidth;

        y += dy;
        if(y < 0)
            y = 0;
        else if(y > windowHeight)
            y = windowHeight;
    }

};

__global__ void rand_init(RNG *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= N) 
        return;

    // TODO: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984, i, 0, &rand_state[i]);
}

__global__ void init_particles(Particle *particles, RNG * rand_state){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    Particle p = particles[i];
    p.pid = i;

    RNG local_rand_state = rand_state[i];

    auto x = curand_uniform_double(&local_rand_state) * double(windowWidth) - 1.0;
    auto y = curand_uniform_double(&local_rand_state) * double(windowHeight) - 1.0;
    p.update(x, y);

    p.vx = curand_uniform_double(&local_rand_state) * 100 - 50.0;
    p.vy = curand_uniform_double(&local_rand_state) * 100 - 50.0;

    rand_state[i] = local_rand_state;
    particles[i] = p;
}


template <typename RNG>
__global__ void apply_forces(Particle *particles, RNG* rand_state, double sqrt_dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    Particle p = particles[i];
    // Apply drag force
    p.vx -= GAMMA / mass * p.vx * dt;
    p.vy -= GAMMA / mass * p.vy * dt;

    // Apply random force
    RNG local_rand_state = rand_state[i];
    double2 r = curand_uniform2_double(&local_rand_state); 
    p.vx += (r.x  * 2.0 - 1.0) * sqrt_dt;
    p.vy += (r.y  * 2.0 - 1.0) * sqrt_dt;
    rand_state[i] = local_rand_state;
    particles[i] = p;

}

__global__ void update_positions(Particle *particles){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;
        
    Particle p = particles[i];

    // Check for collisions with box boundaries
    if (p.x - RADIUS < 0 || p.x + RADIUS > windowWidth) {
        p.vx *= -1;
    }
    if (p.y - RADIUS < 0 || p.y + RADIUS > windowHeight) {
        p.vy *= -1;
    }
    // Update positions
    p.update(p.vx * dt, p.vy * dt);

    particles[i] = p;

}


int main(){
    const double sqrt_dt = std::sqrt(2.0 * T * GAMMA / mass * dt); // Standard deviation for random force
    std::cout << "sqrt_dt: " << sqrt_dt << "\n";

    const double density = (N * PI * RADIUS* RADIUS) / (windowWidth * windowHeight);
    std::cout << "density: " << density << "\n";

    // Random number generator setup
    RNG *d_rand_states;
    checkCudaErrors(cudaMalloc((void **)&d_rand_states, N*sizeof(RNG)));

    // allocate particles
    Particle *particles;
    checkCudaErrors(cudaMallocManaged((void **)&particles, N * sizeof(Particle)));

    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;
    rand_init<<<nblocks, nthreads>>>(d_rand_states);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Initialize particles
    init_particles<<<nblocks, nthreads>>>(particles, d_rand_states);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // Simulation loop
    int iter = 0;
    while (iter++ < STEPS) {
        apply_forces<<<nblocks, nthreads>>>(particles, d_rand_states, sqrt_dt);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        update_positions<<<nblocks, nthreads>>>(particles);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

}