# Quick Start Guide: VeRL with Atropos Integration

This guide will help you get started with running on-policy Reinforcement Learning (RL) using VeRL, integrated with Nous Research's Atropos framework for environment interactions.

## Prerequisites

1.  **Clone the Repository**:
    If you haven't already, clone your fork of the VeRL repository and navigate into its root directory.
    ```bash
    # git clone <your-fork-url>
    # cd verl-repository-root
    ```

2.  **Install Dependencies**:
    Ensure you have Python 3.10+ installed. Then, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    You might also need development dependencies for some functionalities or tests:
    ```bash
    # pip install -r requirements-dev.txt # If such a file exists
    ```
    This integration uses `torch`, `transformers`, `fastapi`, `uvicorn`, `aiohttp`, `pyyaml`, `tensorboard`, and `psutil` (for some tests). Ensure these are covered by your `requirements.txt`.

3.  **Atropos Environment Scripts**:
    This integration relies on Atropos environment server scripts (e.g., `gsm8k_server.py`). These scripts are typically part of the [Atropos repository](https://github.com/NousResearch/atrolib) or your own custom environment implementations following the Atropos framework. You need to have these scripts accessible in a directory. By default, the launch script (`recipe/atropos/launch_atropos_verl.py`) looks for them in a directory named `environments/` relative to where you run the script. You can specify a different path using the `--env-script-path` argument.

## Configuration

1.  **Locate Example Configuration**:
    An example configuration file is provided at `configs/atropos_example.yaml`. Copy or edit this file for your training run.

2.  **Edit Configuration (`configs/your_config_name.yaml`)**:
    Open your copy of `configs/atropos_example.yaml` (or a new file based on it) and modify the following key fields:

    *   **`model`**: Set this to the HuggingFace model name/path you want to train (e.g., `"NousResearch/Llama-2-7b-hf"`, `"mistralai/Mistral-7B-v0.1"`, or a path to a local model).
        ```yaml
        model: "your-hf-model-name-or-path" # TODO: Replace this
        ```

    *   **`reference_model`** (Optional): If you want to use KL divergence regularization, specify a reference model. If `null` or not provided, KL penalty will be skipped (or ensure `kl_coef` is 0).
        ```yaml
        reference_model: "your-reference-model-name-or-path" # or null
        ```

    *   **`environments`**: List the names of the Atropos environments you want to use. These names must correspond to the environment server scripts available in your environment scripts directory (e.g., `gsm8k` for `gsm8k_server.py`).
        ```yaml
        environments:
          - "gsm8k" # Example, replace with your desired environment(s)
          # - "another_env_name" # Ensure 'another_env_name_server.py' or 'another_env_name.py' exists
        ```
    *   **Server Ports**: Adjust `rollout_server_port` (for Atropos API) and `inference_api_port` (for model inference) if the defaults (8000, 8001) are already in use on your system.
        ```yaml
        rollout_server_port: 8000
        inference_api_port: 8001
        ```

    *   **`use_sglang`**: Set to `true` if you intend to use SGLang as an external inference server (requires SGLang installation and that the launch script fully supports managing it). For initial testing with the embedded server, keep it `false`.
        ```yaml
        use_sglang: false
        ```

    *   **Hyperparameters**: Adjust `lr` (learning rate), `batch_size`, `num_iterations`, `max_seq_len`, `ppo_epochs`, `clip_ratio`, `kl_coef`, `entropy_coef`, etc., as needed for your specific model and task.

## Running Training

1.  **Launch the Training Script**:
    From the root directory of the VeRL repository, run the `launch_atropos_verl.py` script, pointing it to your configuration file:
    ```bash
    python recipe/atropos/launch_atropos_verl.py --config configs/your_edited_config.yaml
    ```
    If your Atropos environment scripts are not in an `environments/` subdirectory of your current working directory, specify their location using the `--env-script-path` argument:
    ```bash
    python recipe/atropos/launch_atropos_verl.py \
        --config configs/your_edited_config.yaml \
        --env-script-path /path/to/your/atropos/environment_scripts
    ```

2.  **Dry Run (Optional)**:
    To see a plan of what the script would do without actually starting any servers or the training process, use the `--dry-run` flag:
    ```bash
    python recipe/atropos/launch_atropos_verl.py --config configs/your_edited_config.yaml --dry-run
    ```

## Monitoring and Output

*   **Console Logs**:
    *   The main launch script (`launch_atropos_verl.py`) will print INFO level logs to your console, indicating the status of various components.
*   **Detailed Log Files**:
    *   More detailed logs for each component (inference server if external, Atropos API server, environment servers) are saved into individual files within the `logs/` directory. This directory is created in the location where you run the launch script.
*   **Metrics**:
    *   **TensorBoard**: Training metrics such as losses, rewards, KL divergence, and entropy are logged to TensorBoard. To view these, start TensorBoard by pointing it to the `runs/` directory (also created where you run the launch script):
        ```bash
        tensorboard --logdir runs/
        ```
        Navigate to the URL provided by TensorBoard (usually `http://localhost:6006`). Each training run will create a unique subdirectory within `runs/` (e.g., `runs/grpo_<timestamp>_<model_name>`).
    *   **JSONL File**: The same metrics are also saved in a structured JSONL format to `metrics/logs.jsonl`. Each line in this file is a JSON object representing the metrics for one training iteration. This file is useful for programmatic analysis.

## Stopping Training

*   To stop the training process and all associated servers gracefully, press `Ctrl+C` in the terminal where `launch_atropos_verl.py` is running. The script is designed to catch this interrupt signal and attempt to shut down all components cleanly.

## Next Steps

*   **Implement Custom Environments**: Adapt or create new Atropos environment server scripts for your specific tasks.
*   **Experiment with Models**: Try different base models from HuggingFace or your own pre-trained models.
*   **Tune Hyperparameters**: Adjust the PPO hyperparameters, learning rate, batch size, sequence length, etc., in your YAML configuration file to optimize performance for your application.
*   **Analyze Results**: Use TensorBoard and the JSONL metrics to monitor training progress, diagnose issues, and compare different runs.
*   **Extend Functionality**: Explore and extend the VeRL trainer and associated components for more advanced RL techniques or custom behaviors.

```
