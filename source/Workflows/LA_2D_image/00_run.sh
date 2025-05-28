#!/bin/sh
python 01_generate_model.py -i ./data -r ./result
python 02_run_limit_analysis.py -i ./data -r ./result
python 03_plot_failure_mode.py -i ./data -r ./result
