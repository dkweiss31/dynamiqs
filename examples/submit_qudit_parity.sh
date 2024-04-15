#!/bin/bash
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --job-name=qudit_parity
#SBATCH -o out/output-%a.txt -e out/errors-%a.txt
#SBATCH --array=0-4 ### NUM_ARRAY_PTS - 1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.weiss@yale.edu

################################## modify params here
dts=(10 20 50 100 200)
time=1000.0

gate="parity"
c_dim_1=4
c_dim_2=8
EJ_1=12.606
EJ_2=30.0
EC_1=0.270
EC_2=0.110
g=0.006
dt=${dts[${SLURM_ARRAY_TASK_ID}]}
ramp_nts=2
scale=0.0001
learning_rate=0.001
b1=0.999
b2=0.999
coherent=1
epochs=1000
target_fidelity=0.9995
rng_seed=$((868726 * SLURM_ARRAY_TASK_ID + 3562))

module load miniconda
conda activate dynamiqs
python run_second_order_bin_parity.py \
  --idx=$SLURM_ARRAY_TASK_ID \
  --gate=$gate \
  --c_dim_1=$c_dim_1 \
  --c_dim_2=$c_dim_2 \
  --EJ_1=$EJ_1 \
  --EJ_2=$EJ_2 \
  --EC_1=$EC_1 \
  --EC_2=$EC_2 \
  --g=$g \
  --dt=$dt \
  --time=$time \
  --ramp_nts=$ramp_nts \
  --scale=$scale \
  --learning_rate=$learning_rate \
  --b1=$b1 \
  --b2=$b2 \
  --coherent=$coherent \
  --epochs=$epochs \
  --target_fidelity=$target_fidelity \
  --rng_seed=$rng_seed
