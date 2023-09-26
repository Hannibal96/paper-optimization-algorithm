#!/bin/bash

python='/home/neria/PycharmProjects/Test/venv/bin/python3'

for n in 1000 # 100 
do
	for r in {2..10..2}
	do
		$python multi_lr_sol.py --N ${n} --runs ${r} --data syn --o -n > noise-syn_N=${n}_R=${r}.log & 
		$python multi_lr_sol.py --N ${n} --runs ${r} --data syn  > syn_N=${n}_R=${r}.log &
		$python multi_lr_sol.py --N ${n} --runs ${r} --data mnist > mnist_N=${n}_R=${r}.log  &

	done
done



