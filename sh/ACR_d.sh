#!/bin/bash

version=ACR_d
dataset=$1
idx=$2

lr=0.03
total_steps=250000
ema_u=0.99
tau=2.0
threshold=0.95
mu=1
bz=64
out=out
img_size=32
N_GPU=1
seed=0

if [ "$dataset" = "cifar10" ]; then

	N=(500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500 500)
	M=(4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000 4000)
	gamma_l=(150 150 150 150 150 100 100 100 100 100 150 150 150 150 150 150 150 150 150 100)
	gamma_ul=(150 1 150 150 150 100 1 100 100 100 120 90 60 30 120 90 60 30 150 100)

	reverse=(0 0 1 3 4 0 0 1 3 4 0 0 0 0 1 1 1 1 2 2)

	mark=${version}_${dataset}_N${N[${idx}]}_M${M[${idx}]}_gamma_l${gamma_l[${idx}]}_gamma_ul${gamma_ul[${idx}]}_reverse${reverse[${idx}]}_lr${lr}_bz${bz}_total_steps${total_steps}_threshold${threshold}_seed${seed}_ema_u${ema_u}_mu${mu}_tau${tau}

elif [ "$dataset" = "cifar100" ]; then

	N=(50 50 50 50 50 50 50 50 50 50 50)
	M=(400 400 400 400 400 400 400 400 400 400 400)
	gamma_l=(20 20 20 20 20 10 10 10 10 10 20)
	gamma_ul=(20 1 20 20 20 10 1 10 10 10 20)

	reverse=(0 0 1 3 4 0 0 1 3 4 2)

	mark=${version}_${dataset}_N${N[${idx}]}_M${M[${idx}]}_gamma_l${gamma_l[${idx}]}_gamma_ul${gamma_ul[${idx}]}_reverse${reverse[${idx}]}_lr${lr}_bz${bz}_total_steps${total_steps}_threshold${threshold}_seed${seed}_ema_u${ema_u}_mu${mu}_tau${tau}

elif [ "$dataset" = "stl10" ]; then

	#
	N=(450 450)
	M=(100000 100000)
	gamma_l=(10 20)
	gamma_ul=(10 20) # unused
	reverse=(0 0)    # unused

	mark=${version}_${dataset}_N${N[${idx}]}_M${M[${idx}]}_gamma_l${gamma_l[${idx}]}_gamma_ul${gamma_ul[${idx}]}_reverse${reverse[${idx}]}_lr${lr}_bz${bz}_total_steps${total_steps}_threshold${threshold}_seed${seed}_ema_u${ema_u}_mu${mu}_tau${tau}

elif [ "$dataset" = "smallimagenet" ]; then

	N=(28000 28000)
	M=(250000 250000)
	gamma_l=(286 286)
	gamma_ul=(286 286)
	reverse=(0 0)
	img_sizes=(32 64)
	img_size=${img_sizes[${idx}]}

	mark=${version}_${dataset}_N${N[${idx}]}_M${M[${idx}]}_gamma_l${gamma_l[${idx}]}_gamma_ul${gamma_ul[${idx}]}_reverse${reverse[${idx}]}_lr${lr}_bz${bz}_total_steps${total_steps}_threshold${threshold}_seed${seed}_ema_u${ema_u}_mu${mu}_img_size${img_size}_tau${tau}

elif [ "$dataset" = "smallimagenet_1k" ]; then

	N=(256 256)
	M=(1024 1024)
	gamma_l=(256 256)
	gamma_ul=(256 256)
	reverse=(0 0)
	img_sizes=(32 64)

	img_size=${img_sizes[${idx}]}

	mark=${version}_${dataset}_N${N[${idx}]}_M${M[${idx}]}_gamma_l${gamma_l[${idx}]}_gamma_ul${gamma_ul[${idx}]}_reverse${reverse[${idx}]}_lr${lr}_bz${bz}_total_steps${total_steps}_threshold${threshold}_seed${seed}_ema_u${ema_u}_mu${mu}_img_size${img_size}_tau${tau}

fi

echo start ${mark}
PORT=$(($RANDOM % 1000 + 10000))
echo "PORT=${PORT}"

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} \
	--nnodes=1 --nproc_per_node="${N_GPU}" \
	${version}/train.py \
	--dataset ${dataset} \
	--num-max ${N[${idx}]} \
	--num-max-u ${M[${idx}]} \
	--seed ${seed} \
	--imb-ratio-label ${gamma_l[${idx}]} \
	--imb-ratio-unlabel ${gamma_ul[${idx}]} \
	--ema-u ${ema_u} \
	--flag-reverse-LT ${reverse[${idx}]} \
	--total-steps ${total_steps} \
	--batch-size ${bz} \
	--lr ${lr} \
	--out ${out}/${mark} \
	--threshold ${threshold} \
	--version ${version} \
	--mu ${mu} \
	--num-workers 4 \
	--tau ${tau} \
	--img-size ${img_size}

echo end ${mark}
