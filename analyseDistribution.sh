
#! /bin/bash

# 建立bashlog存在的文件夹
if [ ! -d "bashLogs" ];then
	mkdir bashLogs
else
	echo "bashLogs 文件夹已经存在"
fi

# 构建log文件
REWRITE=false
echo LJCH > tee ./bashLogs/log.txt


# 开启实验循环 
for i in {0..9}
do
	python mainBaseline1.py --testRound $i --verbose 1 | tee -a ./bashLogs/log.txt
done




