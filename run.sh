export PYTHONPATH="/home/neria/Desktop/paper-optimization-algorithm/:$PYTHONPATH"
python='/home/neria/PycharmProjects/Test/venv/bin/python3'

cd mse
${python} algorithm_hpo.py --d 20 --r 3 --m 20 --Tr 100 --s 5 --times 5 --trials 1
${python} test_algorithm.py --opt_d 20 --opt_r 3 --opt_s 5 --tr_opt 100 --test r --min 5 --max 7 --spaces 1 --opt_m 20 --times 5




