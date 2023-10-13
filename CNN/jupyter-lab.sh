#!/bin/bash

###
# Written by CS236781 staff
# used to open jupyter projects on lambda server
#


unset XDG_RUNTIME_DIR

xvfb-run -a -s "-screen 0 1440x900x24" jupyter lab --no-browser --ip=$(hostname -I) --port-retries=100

