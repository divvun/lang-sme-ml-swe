module --quiet purge  # Reset the modules to the system default
module load PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4
module unload PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4
module load CMake/3.15.3-GCCcore-8.3.0

python -m venv env
source env/bin/activate
pip install -r requirements.txt
