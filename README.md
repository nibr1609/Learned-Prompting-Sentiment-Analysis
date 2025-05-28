# Sentiment Classification Project 🌟

## Project Description 📖

Sentiment Classification (SC) is the task of automatically analyzing text to determine its emotional tone, categorizing it as **positive** 😊, **negative** 😠, or **neutral** 😐. This classic Natural Language Processing (NLP) problem remains highly relevant today, powering applications like:

- 📊 Product review analysis
- 🗣️ Social media monitoring
- 💬 Customer feedback interpretation


## Setup Instructions ⚙️

### Creating Conda Environment

To get access to the centralized Conda installation on the Cluster add the following to your ~/.bashrc file and restart your shell to get access to conda:
<pre>
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/cluster/courses/cil/envs/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cluster/courses/cil/envs/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/cluster/courses/cil/envs/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/cluster/courses/cil/envs/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
</pre>

### Create environment and install requirements

Create environment and install requirements via:
`conda env create -f environment.yml`

### Manually rebuild llama-cpp
llama-cpp needs to be manually rebuild to activate CUDA support:

`CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --no-deps --force-reinstall llama-cpp-python==0.3.9`

### Run experiments

To run any experiment from the `CILProject2025/experiments` path,

Add your USERNAME to the model_output_dir in the corresponding config JSON file to run the code

change username in **run_gpu.sh** to your own in the path for **HF_HOME** (line 24)



### Update Requirements (if necessary)

Update requirements:
`conda list --export > requirements.txt`


## Project Structure 🗂️

```
CILProject2025/
├── data/                      # Test and Training data files
├── experiments/               # Configuration files for experiments
├── llms/
├── ...
├── results
├── src/                       # Source code (models, utils, evaluation)
│   ├── ...   
│   ├── llm_building_blocks/    
│   ├── models/                # Model architectures and training scripts
│   ├── utils/                 # Helper functions
│   ├── run_experiments.py
├── submissions/   
├── run_gpu.sh                 # SLURM-compatible experiment runner
├── run_cpu.sh                 # SLURM-compatible experiment runner
├── environment.yml            # Conda environment spec
├── README.md                  # Project documentation
```

Contributers: Niklas Britz, Karl Aurel Deck, Marie Louise Douga, Alexander Zank
