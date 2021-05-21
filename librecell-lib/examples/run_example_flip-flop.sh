#!/bin/bash

# Characterize the INVX1 cell and write the output into invx1.lib.

lctime --liberty template.lib \
    --analyze-cell-function \
    --include gpdk45nm.m \
    --spice DFFPOSX1.pex.netlist \
    --cell DFFPOSX1 \
    --output-loads "0.05" \
    --slew-times "0.1" \
    --related-pin-transition "0.1" \
    --output dffposx1.lib
