# Sentiment Classification Project

## Project Description 📖

Sentiment analysis is the process of determining the general feelings conveyed in a text, categorizing it as **positive** 😊, **negative** 😠, or **neutral** 😐. 
While it may seem straightforward, it poses substantial challenges even for state-of-the-art language models. Emotional tone is often expressed not directly but through context, background knowledge, implications, or stylistic choices — all of which can be subtle or ambiguous.

This complexity is particularly noticeable in short texts, where cues are sparse and interpreting sentiment reliably requires nuanced understanding.

Recent advances in **Large Language Models (LLMs)** have made it possible to approach sentiment classification in novel ways. The field has evolved from:

- Traditional **lexicon-based** or **machine learning** methods like SVM, Naive Bayes
- To **fine-tuned transformer models** like BERT, EmoLLMs
- To **prompt-based approaches**, where language models are treated as black boxes, guided by natural language instructions

These modern methods hinge on *prompt engineering* — the art of crafting inputs that best extract the desired response from a language model. Prompt phrasing can significantly influence performance.

Sentiment analysis remains highly relevant today, powering applications like:

- Product review analysis
- Social media monitoring
- Customer feedback interpretation

## Our Approach 🔧

This project investigates how **prompt quality** and **prompt selection** affect sentiment classification performance. We propose a two-step optimization pipeline:

- **Prompt Selector**
  Dynamically selects the most appropriate prompt from a base catalog based on input features.

- **Prompt Optimizer**
  Analyzes performance trends and generates new high-quality prompts through *meta-prompting*.

Our experiments show that this pipeline improves sentiment accuracy of the **Gemma 3 (4B)** model, compared to static prompts or basic prompt catalogs — *without any model fine-tuning*.


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
llama-cpp needs to be manually rebuilt to activate CUDA support:

`CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir --no-deps --force-reinstall llama-cpp-python==0.3.9`

### Download the recommended model from Hugging Face

We recommend using HF:`google/gemma-3-4b-it-qat-q4_0-gguf` details for download instructions and the termes of use can be found on: `https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf` After the download place the model in the llms folder.

### Run Experiments

All experiments are configured via JSON files located in `CILProject2025/experiments/`.  
You can also create custom tests by writing new `.json` experiment files.

Each config allows you to specify:

- **Model** and its hyperparameters  
- **Experiment settings**, including:
  - `mode`: e.g., `train_inference`, `inference_multiple_prompts`
  - `optimize`: run the full pipeline (prompt selection + optimization)
  - `select`: run only the selection step on the base prompt catalogue
  - `optimize_only`: optimize an already selected set of prompts
- and more

**Note:** Both Hugging Face models used are *discriminative* models.

---

### Before Running

1. Edit the corresponding experiment JSON file and add your username to the `model_output_dir`.
2. In `run_gpu.sh`, change the value of `HF_HOME` (line 24) to reflect your own username.

---

-Use `sbatch run_gpu.sh experiment_name` (omit the .json extension) to run the experiment **on the cluster**.

-Use `python3 run_experiment.py -c experiment_name` (omit the .json extension) to run the experiment **locally**.


## Project Structure 🗂️

```
CILProject2025/
├── data/                      # Test and Training data files
├── experiments/               # Configuration files for experiments
├── llms/                      # Store your llms here (e.g. xyz.gguf)
├── ...
├── results
├── src/                       # Source code (models, utils, evaluation)
│   ├── ...   
│   ├── llm_building_blocks/    
│   ├── models/                # Model architectures and training scripts
│   ├── utils/                 # Helper functions
│   ├── run_experiments.py
├── submissions/               # The submission .csv files are here
├── run_gpu.sh                 # SLURM-compatible experiment runner
├── run_cpu.sh                 # SLURM-compatible experiment runner
├── environment.yml            # Conda environment spec
├── README.md                  # Project documentation
```

Contributers: Niklas Britz, Karl Aurel Deck, Marie Louise Dugua, Alexander Zank
