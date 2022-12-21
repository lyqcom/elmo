
#!/bin/bash
ROOT_PATH=$(pwd)

export RANK_TABLE_FILE=$1
RANK_SIZE=$2

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${ROOT_PATH}/scripts/rank_table_8pcs.json
    export RANK_SIZE=8
    export START_ID=0
}
test_dist_2pcs()
{
    export RANK_TABLE_FILE=${ROOT_PATH}/scripts/rank_table_2pcs.json
    export RANK_SIZE=2
    export START_ID=4
}

test_dist_${RANK_SIZE}pcs

for((i=0;i<${RANK_SIZE};i++));
do
    rm ${ROOT_PATH}/device$i/ -rf
    mkdir ${ROOT_PATH}/device$i
    cd ${ROOT_PATH}/device$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$i
    python ${ROOT_PATH}/train.py  --device_num=$RANK_SIZE >log$i.log 2>&1 &
done
