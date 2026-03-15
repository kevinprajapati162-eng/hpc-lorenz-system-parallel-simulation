# HPC Lorenz System Parallel Simulation

This project evaluates multi-process parallelism for the numerical simulation of the Lorenz system using Python. The Lorenz system, a classic chaotic dynamical system, is solved using the fourth-order Runge-Kutta (RK4) method.

## Project Overview
The project compares sequential and parallel implementations of Lorenz system simulation to study runtime, speedup, and numerical accuracy. It treats each Lorenz system as an independent task, making the workload suitable for parallel execution.

## Technologies Used
- Python
- Multiprocessing
- Numerical Simulation
- RK4 Method
- Matplotlib
- High Performance Computing Concepts

## Key Features
- Sequential simulation of Lorenz systems
- Parallel simulation using Python multiprocessing
- Reusable process pools for better efficiency
- Runtime comparison for different workloads
- Speedup analysis using multiple processes
- Accuracy verification between sequential and parallel results
- Runtime scaling visualisation

## Files Included
- `Lorenz.py` – Python source code
- `HPC Report.pdf` – project report

## Purpose
This project demonstrates practical skills in parallel computing, scientific simulation, performance analysis, and numerical accuracy verification in Python.
