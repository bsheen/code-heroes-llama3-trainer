## Overview

The [Code Heroes](https://www.codeheroes.com.au/) Llama3 Trainer aims to provide a CLI interface to orchestrate the fine-tuning of open source AI models, such as Llama3, using 3rd party services, such as [Lambda Cloud](https://lambdalabs.com/), [Hugging Face](https://huggingface.co/) and [Weights & Biases](https://wandb.ai).

## Features

The CLI provides commands to:
- Launch an instance.
- Transfer files to the instance.
- Apply configuration to the instance.
- Install dependances.
- Run the training script.
- Terminate the instance.

The training script:
- Pulls the training data from Hugging Face.
- Pulls the base model data from Hugging Face.
- Fine-tunes the base model with the training data.
- Pushes the fine-tuned model to Hugging Face.
- Logs the run to Weights & Biases.

## Getting Started

### Install 
Run `npm install -g`.

### Help
Run `trainer --help` for a list of commands & options.

### Configuration

To avoid passing sensitive information via CLI command line set options (or any options) via `./config/trainer.json` 

e.g. 
```
{
    "lambda-cloud-ssh-identity-file":"~/.ssh/lambda_cloud",
    "lambda-cloud-api-key": "secret_brendts-macbook-pro_be5152ed4tdc44a8bbe41e70r4cfe7f1.QB5s12jG80Ngii3We9Lc7tdshruARJd9",
    "lambda-cloud-ssh-key-name": "Brendt's Macbook Pro",
    "lambda-cloud-instance-name-prefix": "bs",
    "weights-and-biases-api-key":"aa5c3276d89211f08300f5615ga11fe637f4897f",
    "hugging-face--access-token":"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"
}
```

Note: 
- Run `git update-index --assume-unchanged config/trainer.json` to stop tracking changes.

## Tech Stack

The following 3rd party services and applications are used to:
- serve datasets and models
- store models
- track experiments
- run models (locally on your dev machine)

### Services
- [Lambda Labs](https://lambdalabs.com/) provides cloud hosted GPU for AI training and inference. 
- [Hugging Face](https://huggingface.co/) is a platform for building, sharing, and collaborating on machine learning (ML) models, datasets, and applications.
- [Weights & Biases](https://wandb.ai) is a platform designed for machine learning (ML) practitioners to track, visualize, and manage ML experiments.

### Applications
- [Ollama](https://ollama.com/) is an open-source project that serves as a powerful and user-friendly platform for running LLMs on your local machine.
- [Enchanted LLM](https://apps.apple.com/us/app/enchanted-llm/id6474268307) provides a user interface, similar to ChatGPT, facilitates chat with models served by Ollama.

## Training
The training script (train.py) is based on the [Unsloth](https://unsloth.ai/) as described in their [documentation](https://github.com/unslothai/unsloth?tab=readme-ov-file#-documentation).

Prior to using Unsloth, frequent out of memory exceptions occured as training quickly used up the available GPU memory. The Unsloth approach provides a pathway to training confidently on limited GPU memory via 4bit quantization. 

Furthermore, Unsloth provides a quantized 4bit version of Meta's [Llama3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) base model - [unsloth/llama-3-8b-Instruct-bnb-4bit](https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit). Utilising Unsloths quantized version as the base model removes the quantization process as a factor which could negatively impact training outcomes. That is, we assume Unsloth provides a correctly quantized verson of the model. 

Warning: 
At time of writing (May 2024) the training script:
- is tested on Meta's [Llama3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model only.
- assumes training data is in chat format.

That is, you can confidently train on Llama3 Instruct based models. 

If you want to train other models please discuss with Brendt on a suggested approach.

## Training Data

The training script expects the training data is:
- pulled from a Dataset on Hugging Face. 
- on JSONL format (i.e each line a record and is valid JSON)
- split for training, testing and validation
- in a specific format (see below)

The following example shows two `chats` between the role `user` and `assistant`. In each example the `user` asks a question and the `assistant` answers the question. 
```
{"chats":[{"role":"user","content":"What is the team's favourite animal?"},{"role":"assistant","content":"I can, with almost certainty, say it is dogs."}]}
{"chats":[{"role":"user","content":"Does the team like cats?"},{"role":"assistant","content":"No so much. They seem to prefer dogs."}]}
```

Note:
- The keys `chats`, `role` and `content` and values `user` and `assistant` are expected by the training script. Please transform your training data to this (standard) format to avoid changing the script unnecessarily.
- The `tokenizer` in the training script transforms the data set into the format required by the model. See [Llama 3 Model Cards & Prompt formats](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/) for more into.
- Despite Meta stating [here](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/) to use `<|end_of_text|>` as the end of sentence special token which "on generating this token, Llama 3 will cease to generate more tokens". At the time of writing (May 2024) there was some confusion in the community as to whether to use `<|end_of_text|>` or `<|eot_id|>`. Whether correct or not, the script explicitly sets `<|eot_id|>` as the eos token, as without this set the fine-tuned model would not cease generation. **This should be revisited  to ensure we are following best practices.**

## Contributions Welcome

Thank you for considering contributing!

1. **Fork the Repository**: Click the 'Fork' button at the top right of this page to create a copy of this repository under your GitHub account.
2. **Clone Your Fork**: Use `git clone` to clone your fork to your local machine.
3. **Install Dependencies**: Run `npm install` to install the necessary dependencies.
4. **Create a Branch**: Create a new branch for your feature or bug fix.
5. **Make Your Changes**: Make your changes in your local repository.
6. **Submit a Pull Request**: Push your branch to GitHub and submit a pull request to the main repository.

## Need Help?

If you have any questions please open an issue.