import streamlit as st
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
FILE_PATH = "/home/ec2-user/training/dev-lora-dir/training_params_and_metrics_global0.jsonl"

def initialize_session_state() -> None:
    """Initialize the session state variables if they don't exist."""
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
    
    if 'file_content' not in st.session_state:
        st.session_state.file_content = "" 
    
    if 'last_modified' not in st.session_state:
        st.session_state.last_modified = 0

    if 'file_path' not in st.session_state:
        st.session_state.file_path = FILE_PATH

def update_file_content() -> None:
    """Read and update the file content in session state."""
    try:
        with open(st.session_state.file_path, 'r') as file:
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
        "File Content",
        value=st.session_state.file_content,
        height=400,
        disabled=True
    )
    # Show last update time
    if st.session_state.last_modified:
        st.text(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state.last_modified))}")



    progress_bar = st.progress(0)
    status_text = st.empty()
    return progress_bar, status_text

def handle_user_controls() -> None:
    """Handle user interface controls and their interactions."""
    col1 = st.columns(1)[0]
    with col1:
        if st.button("Start Training"):
            st.session_state.training_started = True
        
        if st.button("Reset"):
            st.session_state.training_started = False
            st.rerun()

def fake_training(progress_bar: st.container, status_text: st.container) -> None:
    """
    Simulate a training process with progress updates.
    
    Args:
        progress_bar: Streamlit progress bar component
        status_text: Streamlit text component for status updates
    """
    try:
        for i in range(PROGRESS_STEPS):
            if not st.session_state.training_started:
                progress_bar.progress(0)
                break
                
            progress_bar.progress(i)
            status_text.text(f"Training progress: {i}%")
            time.sleep(SIMULATION_DELAY)
            
        if st.session_state.training_started:
            st.success("Training completed!")
            
    except Exception as e:
        st.error(f"An error occurred during training: {str(e)}")

def main():
    """Main application entry point."""
    initialize_session_state()
    progress_bar, status_text = create_page_layout()
    handle_user_controls()
        
    # # Start file monitoring if not already started
    if 'observer' not in st.session_state:
        setup_file_monitoring()

    update_file_content()
    time.sleep(UPDATE_INTERVAL)
    
    if st.session_state.training_started:
        fake_training(progress_bar, status_text)
    st.rerun()

if __name__ == "__main__":
    main()

# import streamlit as st
# import time
# from typing import Tuple
# from pathlib import Path
# import asyncio
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler, FileModifiedEvent

# # Constants
# UPDATE_INTERVAL = 1  # seconds
# DEFAULT_FILE_PATH = "path/to/your/file.txt"  # Change this to your file path

# def initialize_session_state() -> None:
#     """Initialize the session state variables."""
#     if 'file_content' not in st.session_state:
#         st.session_state.file_content = ""
#     if 'last_modified' not in st.session_state:
#         st.session_state.last_modified = 0

# class FileChangeHandler(FileSystemEventHandler):
#     """Handle file system events for the monitored file."""
#     def on_modified(self, event: FileModifiedEvent):
#         if event.src_path == str(Path(st.session_state.file_path).resolve()):
#             update_file_content()

# def update_file_content() -> None:
#     """Read and update the file content in session state."""
#     try:
#         with open(st.session_state.file_path, 'r') as file:
#             st.session_state.file_content = file.read()
#             st.session_state.last_modified = time.time()
#     except Exception as e:
#         st.error(f"Error reading file: {str(e)}")

# def setup_file_monitoring() -> None:
#     """Set up the file system observer for monitoring changes."""
#     file_path = Path(st.session_state.file_path)
#     observer = Observer()
#     observer.schedule(FileChangeHandler(), str(file_path.parent), recursive=False)
#     observer.start()
#     st.session_state.observer = observer

# def create_page_layout() -> None:
#     """Create the main page layout."""
#     st.title("File Monitor")
    
#     # File path input
#     file_path = st.text_input(
#         "Enter file path to monitor",
#         value=DEFAULT_FILE_PATH,
#         key="file_path"
#     )
    
#     # Display file content in a scrollable text area
#     st.text_area(
#         "File Content",
#         value=st.session_state.file_content,
#         height=400,
#         disabled=True
#     )
    
#     # Show last update time
#     if st.session_state.last_modified:
#         st.text(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state.last_modified))}")

# def main():
#     """Main application entry point."""
#     initialize_session_state()
#     create_page_layout()
    
#     # Start file monitoring if not already started
#     if 'observer' not in st.session_state:
#         setup_file_monitoring()
    
#     # Update content periodically
#     update_file_content()
#     time.sleep(UPDATE_INTERVAL)
#     st.rerun()

# if __name__ == "__main__":
#     main()