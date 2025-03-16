#!/bin/bash
#SBATCH --job-name=llama3_sent
#SBATCH --mail-user="k.j.oostrum@umail.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-medium
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --gres=gpu:a100:1

module purge

module load ALICE/default
module load Python/3.11.5-GCCcore-13.2.0
export LD_LIBRARY_PATH=/easybuild/software/Python/3.11.5-GCCcore-13.2.0/lib:$LD_LIBRARY_PATH
module load CUDA/12.1.1



echo "## Starting llama test on $HOSTNAME"

SCRATCH_DIR="/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}"
echo "Using local scratch $SCRATCH_DIR"

cp $HOME/data1/llama2_test/llama3_speech_sent2.json $SCRATCH_DIR/
cp $HOME/data1/llama2_test/test_transcripts.txt $SCRATCH_DIR/
echo "Input files copied to scratch"

source ./llama2_test_venv/bin/activate
echo "Virtual env loaded"

export HF_HOME="$SCRATCH_DIR/huggingface_cache"
mkdir -p "$HF_HOME"
echo "Set Hugging Face cache to $HF_HOME"
export TRITON_CACHE_DIR="$SCRATCH_DIR/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

export PYTHONUNBUFFERED=1

cd "$SCRATCH_DIR"
choicellm test_transcripts.txt --model "unsloth/llama-3-70b-bnb-4bit" --prompt llama3_speech_sent2.json --mode categorical > results.csv
echo "Results gotten"

cp results.csv $HOME/data1/llama2_test/
