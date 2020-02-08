#!/bin/bash -e
set -x

# Instrument everything!
tau_instrumentor example.pdb example.cpp -o example.inst.cpp -inline
# Don't instrument inline functions
tau_instrumentor example.pdb example.cpp -o example.inst.cpp -noinline
# Do instrument inline functions, but not short functions
tau_instrumentor example.pdb example.cpp -o example.inst.cpp -inline -minsize 5