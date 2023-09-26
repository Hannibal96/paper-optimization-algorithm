#!/bin/bash

python='/home/neria/PycharmProjects/Test/venv/bin/python3'
#$python algorithm_hpo.py --d 20 --r 3 --m 20 --T 100 --times 10 --sigma 1 --trials 100 > logger.log &


$python test_algorithm.py --opt_d 20 --opt_r 3 --opt_m 20 --opt_tr 100 --opt_times 10 --opt_sigma 1 --test m > logger.log &




