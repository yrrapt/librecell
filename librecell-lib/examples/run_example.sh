#!/bin/bash

# Abort on errors.
set -e

# Characterize the INVX1 cell and write the output into invx1.lib.

# Guide the characterization based on a template liberty file which contains a specification of the cell pins.
lctime --liberty invx1_template.lib \
    --include gpdk45nm.m \
    --spice INVX1.pex.netlist \
    --cell INVX1 \
    --output-loads "0.05, 0.1, 0.2, 0.4, 0.8, 1.6" \
    --slew-times "0.1, 0.2, 0.4, 0.8, 1.6, 3.2" \
    --output invx1.lib


# Here the template liberty file does NOT contain a cell specification.
# With the --analize-cell-function flag the cell specification will be infered from the netlist.
lctime --liberty template.lib \
    --analyze-cell-function \
    --include gpdk45nm.m \
    --spice AND2X1.pex.netlist \
    --cell AND2X1 \
    --output-loads "0.05, 0.1, 0.2, 0.4, 0.8, 1.6" \
    --slew-times "0.1, 0.2, 0.4, 0.8, 1.6, 3.2" \
    --output and2x1.lib
