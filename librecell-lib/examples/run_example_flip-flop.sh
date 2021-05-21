#!/bin/bash

# Characterize the DFFSR flip-flop (with set and reset).

lctime --liberty template.lib \
    --analyze-cell-function \
    --include gpdk45nm.m \
    --spice DFFSR.pex.netlist \
    --cell DFFSR \
    --output-loads "0.05" \
    --slew-times "0.1" \
    --related-pin-transition "0.1" \
    --output dffsr.lib
