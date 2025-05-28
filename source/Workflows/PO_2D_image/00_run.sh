#!/bin/sh
python 01_generate_model.py -i ./examples/example02/data -r ./examples/example02/result
python 02_run_pushover.py -i ./examples/example02/data -r ./examples/example02/result
python 03_plot_pushover_curve.py -i ./examples/example02/data -r ./examples/example02/result
