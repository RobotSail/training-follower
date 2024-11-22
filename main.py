import streamlit as st
import subprocess
import json
import os

FILE_PATH = "/home/ec2-user/experiments/training_output/training_params_and_metrics_global0.jsonl"
PYTHON_INTERPRETER = "/home/ec2-user/training/venv/bin/python3"
FULL_TRAINING_CMD = """--nnodes=1 --node_rank=0 --nproc_per_node=4 --rdzv_id=123 --rdzv_endpoint=0.0.0.0:8888 /home/ec2-user/training/src/instructlab/training/main_ds.py --model_name_or_path=/home/ec2-user/.cache/huggingface/hub/models--ibm-granite--granite-3.0-8b-base/snapshots/23357b69523bd98523496a5aba1f48bdea04a137 --data_path=/home/ec2-user/data/data.jsonl --output_dir=/home/ec2-user/experiments/training_output --num_epochs=2 --effective_batch_size=128 --learning_rate=2e-05 --num_warmup_steps=25 --save_samples=0 --log_level=INFO --max_batch_len=1024 --seed=42 --chat-tmpl-path=/home/ec2-user/training/src/instructlab/training/chat_templates/ibm_generic_tmpl.py --checkpoint_at_epoch --distributed_training_framework=fsdp --cpu_offload_optimizer --cpu_offload_optimizer_ratio=1.0 --cpu_offload_optimizer_pin_memory --cpu_offload_params_fsdp --fsdp_sharding_strategy=SHARD_GRAD_OP"""
TORCHRUN = "/home/ec2-user/training/venv/bin/torchrun"
SCRIPT = "/home/ec2-user/training/super-cool-test-3.py"


def start_training_job():
    # training_command = [PYTHON_INTERPRETER, SCRIPT]
    training_command = [
        TORCHRUN,
        "--nnodes=1",
        "--node_rank=0",
        "--nproc_per_node=4",
        "--rdzv_id=123",
        "--rdzv_endpoint=0.0.0.0:8888",
        "/home/ec2-user/training/src/instructlab/training/main_ds.py",
        "--model_name_or_path=/home/ec2-user/.cache/huggingface/hub/models--ibm-granite--granite-3.0-8b-base/snapshots/23357b69523bd98523496a5aba1f48bdea04a137",
        "--data_path=/home/ec2-user/data/data.jsonl",
        "--output_dir=/home/ec2-user/experiments/training_output",
        "--num_epochs=1",
        "--effective_batch_size=128",
        "--learning_rate=2e-05",
        "--num_warmup_steps=25",
        "--save_samples=0",
        "--log_level=INFO",
        "--max_batch_len=1024",
        "--seed=42",
        "--chat-tmpl-path=/home/ec2-user/training/src/instructlab/training/chat_templates/ibm_generic_tmpl.py",
        "--checkpoint_at_epoch",
        "--distributed_training_framework=fsdp",
        "--cpu_offload_optimizer",
        "--cpu_offload_optimizer_ratio=1.0",
        "--cpu_offload_optimizer_pin_memory",
        "--cpu_offload_params_fsdp",
        "--fsdp_sharding_strategy=SHARD_GRAD_OP",
    ]
    # process = subprocess.Popen(training_command)

    # Open files to capture stdout and stderr
    stdout_file = open("training_stdout.log", "w")
    stderr_file = open("training_stderr.log", "w")

    # Start the training job and capture stdout and stderr
    process = subprocess.Popen(
        training_command,
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,  # Ensure text mode for strings
    )
    st.session_state.training_process = process


def get_training_progress():
    # log_file_path = "/path/to/your/checkpoints_dir/training_log.jsonl"

    if not os.path.exists(FILE_PATH):
        return 0.0

    with open(FILE_PATH, "r") as f:
        lines = f.read().splitlines()

    if len(lines) < 2:
        return 0.0

    print(f"lines length: {len(lines)}")
    first_log, last_log = lines[0], lines[-1]
    first_log, last_log = json.loads(first_log), json.loads(last_log)

    # make sure that `last_log` actually contains the data we need to begin computing progress:
    required_keys = ["epoch", "total_samples", "samples_seen"]
    if not all(key in last_log for key in required_keys):
        print("the last log doesn't yet contain the keys we require")
        return 0.0

    # Here we compute the total progress. The data to compute the progress is fragmented,
    # so here are the details:
    # 1. We get the total number of epochs we will iterate through from the first log output (when we log script args)
    #       - `num_epochs`: This is a 1-indexed value representing the number of times we'll iterate through the dataset
    # 2. Get the information about state of training from the last log output (will be the most recent in the running log output)
    #    From here we get:
    #       - `samples_seen`: Total number of samples seen thus far
    #       - `epoch`: The current epoch, as a 0-indexed value
    #       - `total_samples`: Total number of samples in the overall dataset
    # 3. Our progress therefore is: % done = (completed_epochs * total_samples + samples_seen % total_samples) / (total_epochs * total_samples)
    #
    # IMPORTANT: The `num_epochs` value is 1-indexed, while `epoch` is 0-INDEXED. So `epoch == 0` means we are on epoch 1.
    num_epochs = first_log["script_params"]["num_epochs"]
    current_epoch = last_log["epoch"]
    total_samples = last_log["total_samples"]
    samples_seen = last_log["samples_seen"]
    progress = (current_epoch * total_samples + samples_seen % total_samples) / (
        num_epochs * total_samples
    )
    print(f"{progress=}")
    return min(progress, 1.0)


@st.experimental_fragment(run_every=1)
def main():
    st.title("Training Monitor")

    if "training_in_progress" not in st.session_state:
        st.session_state.training_in_progress = False

    training_btn_label = (
        "Stop Training" if st.session_state.training_in_progress else "Start Training"
    )
    if st.button(training_btn_label, key="toggle-training"):
        if not st.session_state.training_in_progress:
            start_training_job()
            st.session_state.training_in_progress = True
        else:
            st.warning("Training is already in progress.")

    if st.session_state.training_in_progress:
        progress = get_training_progress()
        print(f"training progress: {progress:.4f}")
        progress_bar = st.progress(progress)
        st.text(f"Training Progress: {progress * 100:.2f}%")

        # Check if the process has terminated
        if st.session_state.training_process.poll() is not None:
            st.success("Training Complete!")
            st.session_state.training_in_progress = False
            st.session_state.training_process = None
    else:
        st.info("Click 'Start Training' to begin.")


if __name__ == "__main__":
    main()
