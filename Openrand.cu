#include <iostream>
#include <cmath>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "phillox.h"

#define FQUALIFIER __host__ __device__

#define PI           3.14159265358979323846 
#define GLOBAL_SEED 0x43

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

typedef Phillox RNG;


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

    FQUALIFIER Particle(double x, double y) : x(x), y(y) 
    {

    }

    FQUALIFIER void update(double dx, double dy) {
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


__global__ void init_particles(Particle *particles, int counter){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    Particle p = particles[i];
    p.pid = i;

    RNG local_rand_state(p.pid, counter, GLOBAL_SEED);

    auto x = local_rand_state.rand<double>() * double(windowWidth) - 1.0;
    auto y = local_rand_state.rand<double>() * double(windowHeight) - 1.0;
    p.update(x, y);

    p.vx = local_rand_state.rand<double>() * 100 - 50.0;
    p.vy = local_rand_state.rand<double>() * 100 - 50.0;

    particles[i] = p;
}


__global__ void apply_forces(Particle *particles, int counter, double sqrt_dt){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N)
        return;

    Particle p = particles[i];
    // Apply drag force
    p.vx -= GAMMA / mass * p.vx * dt;
    p.vy -= GAMMA / mass * p.vy * dt;

    // Simulate collision using linear spring forces
    const double k = 2.0; 

    for (auto j : {-1, 1}) {
        j = (i + N + j) % N;  
        Particle q = particles[j];
        double dx = p.x - q.x;
        double dy = p.y - q.y;
        double dist = sqrt(dx * dx + dy * dy);

        if (dist < 4 * RADIUS) {  
            double force = -k * (dist - 2 * RADIUS);  
            double force_x = force * dx / dist;  
            double force_y = force * dy / dist;

            p.vx += force_x * dt;  
            p.vy += force_y * dt;
        }
    }

    // Apply random force
    RNG local_rand_state(p.pid, counter);
    
    //double2 r = curand_uniform2_double(&local_rand_state); 
    auto x = local_rand_state.rand<double>();
    auto y = local_rand_state.rand<double>();
    p.vx += (x  * 2.0 - 1.0) * sqrt_dt;
    p.vy += (y  * 2.0 - 1.0) * sqrt_dt;
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


void test(Particle *particles, int nthreads){
    const double sqrt_dt = std::sqrt(2.0 * T * GAMMA / mass * dt); // Standard deviation for random force
    std::cout << "sqrt_dt: " << sqrt_dt << "\n";

    const double density = (N * PI * RADIUS* RADIUS) / (windowWidth * windowHeight);
    std::cout << "density: " << density << "\n";


    const int nblocks = (N + nthreads - 1) / nthreads;

    // Initialize particles
    init_particles<<<nblocks, nthreads>>>(particles, 0);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // Simulation loop
    int iter = 0;
    while (iter++ < STEPS) {
        apply_forces<<<nblocks, nthreads>>>(particles, iter, sqrt_dt);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        update_positions<<<nblocks, nthreads>>>(particles);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }


}

int main(int argc, char **argv) {
    // allocate particles
    Particle *particles;
    checkCudaErrors(cudaMallocManaged((void **)&particles, N * sizeof(Particle)));

    bool benchmark = false;

    if(benchmark){
        // benchmark
        test(particles, 256);
        return 0;
    }
    else{
        // Reproducibility check
        test(particles, 256);
        Particle *particles2;
        checkCudaErrors(cudaMallocManaged((void **)&particles2, N * sizeof(Particle)));
        test(particles2, 512);

        // Reproducibility check
        std::vector<double> errors (N*2);
        for(int i=0; i<N; i++){
            errors[i*2] = particles[i].x - particles2[i].x;
            errors[i*2+1] = particles[i].y - particles2[i].y;
        }

        // To reduce floating point errors
        std::sort(errors.begin(), errors.end());
        
        // average error
        double avg = 0;
        for(int i=0; i<N; i++){
            avg += errors[i];
        }
        std::cout << "Total error: " << avg << "\n";
        cudaFree(particles2);
    }

    cudaFree(particles);
    return 0;
}