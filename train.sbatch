#!/usr/bin/env bash
#SBATCH --mem  60GB
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 16
#SBATCH --constrain smaug
#SBATCH --mail-type FAIL BEGIN END
#SBATCH --mail-user sanazsab@kth.se
#SBATCH --output /Midgard/home/sanazsab/Detailed-virtual-try-on/logs/J_slurm.out
#SBATCH --error  /Midgard/home/sanazsab/Detailed-virtual-try-on/logs/J_slurm.err


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate DVTO
echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS}"
#python train.py --train_mode gmm --batch_size_t 64
#python train.py --train_mode parsing
#python train.py --train_mode depthi
#CUDA_VISIBLE_DEVICES=0 python demo.py --batch_size_v 1 --num_workers 4 --warp_c
#python train.py --train_mode parsing
python train.py --train_mode appearance --batch_size_t 6 --save_time --gpu_ids_n 2 --suffix refactornew --val_freq 20 --save_epoch_freq 1 --joint_all #--joint_parse_loss
#python train.py --train_mode depth --add_grad_loss --batch_size_t 8 #--add_gan_loss #--add_normal_loss     #smaug, shelob for local_storage
#python train.py --train_mode face --batch_size_t 16
#python train.py --train_mode appearance --batch_size_t 4 --joint_all --joint_parse_loss --gpu_ids_n 2
