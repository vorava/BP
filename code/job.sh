#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=2:ngpus=2:mem=75gb:scratch_local=10gb:gpu_cap=cuda61:pbs_server=meta-pbs.metacentrum.cz:cluster=galdor
#PBS -N trainModel_resnet
#PBS -m ae

DATADIR=/storage/brno2/home/xorava02
cd $DATADIR

mkdir job$PBS_JOBID


MISTO_INSTALACE=/storage/brno2/home/xorava02/obj_detect
DATADIR=$DATADIR/job$PBS_JOBID
cd $MISTO_INSTALACE

export TMPDIR=$SCRATCHDIR

eval "$(./bin/micromamba shell hook -s bash)"
MAMBA_EXE=bin/micromamba micromamba activate $(pwd)/tf2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/storage/brno2/home/xorava02/obj_detect/tf2/lib
export TF_XLA_FLAGS="--tf_xla_auto_jit=2"


python TensorFlow/models/research/object_detection/model_main_tf2.py --model_dir=TensorFlow/workspace/models/ssd_resnet_normal --pipeline_config_path=TensorFlow/workspace/models/ssd_resnet_normal/pipelineT.config --num_workers=6

clean_scratch
exit 0
