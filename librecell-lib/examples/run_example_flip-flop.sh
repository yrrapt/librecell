#!/bin/bash

# Characterize the INVX1 cell and write the output into invx1.lib.

lctime --liberty dffposx1_template.lib \
    --include gpdk45nm.m \
    --spice DFFPOSX1.pex.netlist \
    --cell DFFPOSX1 \
    --output-loads "0.05, 0.1, 0.2, 0.4, 0.8, 1.6" \
    --slew-times "0.1, 0.2, 0.4, 0.8, 1.6, 3.2" \
    --output dffposx1.lib
