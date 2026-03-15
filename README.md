**HPC Lorenz System Parallel Simulation**

This project evaluates multi-process parallelism for the numerical simulation of the Lorenz system using Python. The Lorenz system, a classic chaotic dynamical system, is solved using the fourth-order Runge-Kutta (RK4) method. The project compares sequential and parallel implementations to measure runtime, speedup, and numerical accuracy.

**Project Overview**

The main goal of this project is to study how multiprocessing can improve performance when simulating many independent Lorenz systems. Each system is treated as an independent task, making the workload suitable for parallel execution.

**Technologies Used**

Python

Multiprocessing

Numerical Simulation

RK4 Method

Matplotlib

High Performance Computing Concepts

**Key Features**

Sequential simulation of Lorenz systems

Parallel simulation using Python multiprocessing

Reusable process pools for improved efficiency

Runtime comparison for different workload sizes

Speedup analysis using 2 and 4 processes

Accuracy verification between sequential and parallel results

Runtime scaling visualisation

**Files Included**

Lorenz.py – Python source code

HPC Report.pdf – project report

**Results Summary**

The project shows that parallel execution can significantly reduce runtime as the number of simulated systems increases, while maintaining numerical accuracy.
