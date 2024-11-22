import json
import os
import signal
import streamlit as st
import subprocess
import time
from typing import Tuple
from pathlib import Path
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent


# Constants
SIMULATION_DELAY = 0.1
PROGRESS_STEPS = 101
UPDATE_INTERVAL = 10
FILE_PATH = "/home/ec2-user/experiments/training_output/training_params_and_metrics_global0.jsonl"
PYTHON_INTERPRETER = "/home/ec2-user/training/venv/bin/python3"


def initialize_session_state() -> None:
    """Initialize the session state variables if they don't exist."""
    if "training_started" not in st.session_state:
        st.session_state.training_started = False

    if "file_content" not in st.session_state:
        st.session_state.file_content = ""

    if "last_modified" not in st.session_state:
        st.session_state.last_modified = 0

    if "file_path" not in st.session_state:
        st.session_state.file_path = FILE_PATH


def update_file_content() -> None:
    """Read and update the file content in session state."""
    try:
        with open(st.session_state.file_path, "r") as file:
            st.session_state.file_content = file.read()
            st.session_state.last_modified = time.time()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")


def setup_file_monitoring() -> None:
    """Set up the file system observer for monitoring changes."""
    file_path = Path(st.session_state.file_path)
    observer = Observer()
    observer.schedule(FileChangeHandler(), str(file_path.parent), recursive=False)
    observer.start()
    st.session_state.observer = observer


class FileChangeHandler(FileSystemEventHandler):
    """Handle file system events for the monitored file."""

    def on_modified(self, event: FileModifiedEvent):
        if event.src_path == str(Path(st.session_state.file_path).resolve()):
            update_file_content()


def create_page_layout() -> Tuple[st.container, st.container]:
    """Create and return the main page layout components."""
    st.title("Fake Training Progress Demo")

    # Display file content in a scrollable text area
    st.text_area(
        "File Content", value=st.session_state.file_content, height=400, disabled=True
    )
    # Show last update time
    if st.session_state.last_modified:
        st.text(
            f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state.last_modified))}"
        )

    progress_bar = st.progress(0)
    status_text = st.empty()
    return progress_bar, status_text


def start_training():
    proc = subprocess.Popen(
        [PYTHON_INTERPRETER, "/home/ec2-user/training/super-cool-test-3.py"],
    )
    st.session_state.file_content = ""
    st.session_state.training_started = True
    st.session_state.training_pid = proc.pid
    st.session_state.training_proc = proc

    # # Start file monitoring if not already started
    if "observer" not in st.session_state:
        setup_file_monitoring()


def stop_training():
    if not st.session_state.training_started:
        # if stop_training was called then this should not be the case
        # but we will continue anyway because the other things are worth checking regardless
        print("stop_training was called but training_started is false :/")

    try:
        os.kill(st.session_state.training_pid, signal.SIGTERM)
        print(f"Process {st.session_state.training_pid} has been terminated")
    except ProcessLookupError:
        print(f"No process with {st.session_state.training_pid} found")
    except Exception as e:
        print(f"Error terminating process {st.session_state.training_pid}: {str(e)}")
    finally:
        st.session_state.training_pid = None
        st.session_state.training_proc = None
        st.session_state.file_content = ""
        st.session_state.training_started = False


def handle_user_controls() -> None:
    """Handle user interface controls and their interactions."""
    col1 = st.columns(1)[0]
    with col1:
        if st.button("Start Training"):
            start_training()

        if st.button("Reset"):
            stop_training()
            st.rerun()


def observe_training_progress(
    progress_bar: st.container, status_text: st.container
) -> None:
    """
    Simulate a training process with progress updates.

    Args:
        progress_bar: Streamlit progress bar component
        status_text: Streamlit text component for status updates
    """
    try:
        while st.session_state.training_started:
            update_file_content()

            # read the last piece of the log output & parse as json
            lines = st.session_state.file_content.splitlines()
            first_log, last_log = lines[0], lines[-1]
            first_log, last_log = first_log, json.loads(last_log)

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
            num_epochs = first_log["script_args"]["num_epochs"]
            print(f"{num_epochs=}")
            current_epoch = last_log["epoch"] + 1
            print(f"{current_epoch=}")
            total_samples = last_log["total_samples"]
            print(f"{total_samples=}")
            samples_seen = last_log["samples_seen"]
            print(f"{samples_seen=}")
            progress = (
                current_epoch * total_samples + samples_seen % total_samples
            ) / (num_epochs * total_samples)

            progress_bar.progress(progress * 100)
            time.sleep(1)

        if st.session_state.training_started:
            st.success("Training completed!")

    except Exception as e:
        st.error(f"An error occurred during training: {str(e)}")


def main():
    """Main application entry point."""
    initialize_session_state()
    progress_bar, status_text = create_page_layout()
    handle_user_controls()

    if st.session_state.training_started:
        observe_training_progress(progress_bar, status_text)
    st.rerun()


if __name__ == "__main__":
    main()
