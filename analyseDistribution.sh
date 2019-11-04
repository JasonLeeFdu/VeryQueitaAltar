#! /bin/bash


for i in {67..100}
do
	python mainBaseline.py --testRound $i --verbose 0 | tee log.txt
done



