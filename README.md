# Sentiment Classification Project

## Project Description

Sentiment analysis is a well-studied machine learning task that is challenging to solve even for state-of-the-art large language models (LLMs). We propose a learned prompting framework that achieves robust sentiment classification while treating LLMs as black boxes. Our method combines (1) a lightweight prompt selector that chooses optimal prompts from a fixed catalog based on input characteristics, and (2) an iterative prompt optimizer that evolves the prompt catalog using meta-prompting techniques.

Experiments demonstrate that our adaptive prompt selection significantly outperforms static prompting baselines, hand-crafted individual prompts, and ensemble voting techniques. While achieving competitive performance with fine-tuned discriminative models like DeBERTa and RoBERTa, our approach only requires training the lightweight selector model, avoiding expensive fine-tuning of the LLM itself. These results confirm that learned prompting is a viable alternative to expensive fine-tuning for sentiment analysis tasks.

## Setup Instructions

### Create environment and install requirements

Create environment and install requirements via:
`conda env create -f environment.yml`

### Manually rebuild llama-cpp
llama-cpp needs to be manually rebuilt to activate CUDA support:

`CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --no-deps --force-reinstall llama-cpp-python==0.3.9`

### Download the recommended model from Hugging Face

In the evaluation of the pipeline, we used the Gamma 3 4b LLM from HuggingFace (`google/gemma-3-4b-it-qat-q4_0-gguf`). Details for download instructions and the termes of use can be found on: `https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf`. After the download place the model in the llms folder.

### Configurations

All experiments are configured via JSON files located in `CILProject2025/experiments/`.  
You can also create custom tests by writing new `.json` experiment files.

Each config allows you to specify:

- **Model settings** and its hyperparameters  
- **Experiment settings**, including:
  - `mode`: e.g., `train_inference`,  `inference`, latter useful for pre-trained HF models
  - `two_stage_setting`: `only_optimize`,  `selection` or `optimize_and_select`
  - `max_train_samples`: number of training samples to use in general
  - `max_train_samples_optimizer`: optionally, we can train optimizer with less samples (as computation can be expensive)
- **Data settings**: Were files are located and where to save results to. 
- **Data settings**, including:
  - `prompt_list`: standard catalog to use
  - `k_best_worst`: how many good and bad prompts to use for prompt optimization
  - `optimizer_iterations`: optimization iterations of optimizer

---

### Before Running

1. You might want to edit the data paths in the config such as `model_output_dir` to fit to your system.
2. In `run_gpu.sh`, you can change the value of `HF_HOME` if desired to load huggingface models to this path. You can also delete the line that exports the environment variable to store models to the standard `.cache` directory.

### Executing Jobs

- Use `sbatch run_gpu.sh experiment_name` (omit the .json extension) to run the experiment **on the cluster**.

- Use `python3 run_experiment.py -c experiment_name` (omit the .json extension) to run the experiment **locally**.


## Project Structure

```
CILProject2025/
├── data/                      # Test and Training data files
├── experiments/               # Configuration files for experiments
├── llms/                      # Store your llms here (e.g. xyz.gguf)
├── misc/                      # Supplementary Files
├── paper/                     # tex code for paper & pdf
├── results/                   # results for experiments
├── src/                       # Source code (models, utils, evaluation)
│   ├── ...   
│   ├── llm_building_blocks/   # Building blocks for pipeline
│   ├── models/                # Model architectures
│   ├── ...                 # Helper functions
│   ├── run_experiments.py
│   ├── launch_tensorboard.py  # Can be used to visualize log of fine-tuning huggingface models
├── submissions/               # The submission .csv files are here
├── run_gpu.sh                 # SLURM-compatible experiment runner
├── run_cpu.sh                 # SLURM-compatible experiment runner
├── environment.yml            # Conda environment spec
├── README.md                  # Project documentation
```

Contributers: Niklas Britz, Karl Aurel Deck, Marie Louise Dugua, Alexander Zank
