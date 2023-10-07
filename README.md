# brownian-dynamics

This is a simplified 2D Brownian dynamics simulation program implemented in CUDA used to benchmark curand and random123 libraries against OpenRAND in Paper "*OpenRAND: A Performance Portable, Reproducible Random Number Generation Library for Parallel Computations*."

One million particles are simulated for 10000 steps. The particles are acted upon by two forces- a drag force proportional to velocity and a random uniform force. The main kernel, `apply_forces`, is quite simple.

Please run `make` to compile the program, making sure to modify the GPU arch beforehand. The `curand`, `r123` and `openrand` executables will be generated.


To check OpenRAND is reproducible, simply run `./openrand` twice. The output should be identical.

```
./openrand > a.txt
./openrand > b.txt
cmp a.txt b.txt
```
