import os
import re
import json
import dxdata
import logging
import subprocess
import pandas as pd
from pathlib import Path


def get_dataset_id(project: str | None = None, full: bool = True) -> str:
    """Return the DNAnexus dataset identifier.

    Args:
        project: DNAnexus project ID. Defaults to DX_PROJECT_CONTEXT_ID.
        full: If True, return "{project}:{record_id}". If False, return project ID.

    Returns:
        Dataset identifier string.

    Raises:
        RuntimeError: If project is not available.
    """
    proj = project or os.environ.get("DX_PROJECT_CONTEXT_ID")
    if not proj:
        raise RuntimeError("Set DX_PROJECT_CONTEXT_ID or pass 'project'.")

    # get first line from dx find output
    line = (
        subprocess.check_output(
            ["dx", "find", "data", "--type", "Dataset", "--brief"], text=True
        )
        .strip()
        .splitlines()[0]
    )

    if not full:
        # just return the project id
        return proj

    # full: if dx output already included the project return it, otherwise prepend proj
    return line if ":" in line else f"{proj}:{line}"



def connect_to_dataset(cohort_record_id: str = ""):
    """Connect to a DNAnexus dataset or cohort.

    Args:
        cohort_record_id: Optional cohort record ID. If empty, loads first dataset.

    Returns:
        Loaded dxdata.Dataset or dxdata.Cohort.

    Raises:
        RuntimeError: If DX_PROJECT_CONTEXT_ID is not set.
    """
    project = os.environ.get("DX_PROJECT_CONTEXT_ID")
    if not project:
        raise RuntimeError("DX_PROJECT_CONTEXT_ID is not set.")

    if cohort_record_id:
        # User explicitly requested a cohort
        dataset_id = f"{project}:{cohort_record_id}"
        return dxdata.load_cohort(id=dataset_id)
    else:
        # Default: whole cohort dataset
        dataset_id = get_dataset_id(project=project)
        return dxdata.load_dataset(id=dataset_id)


def find_latest_dx_file_id(name_pattern: str, folder: str = None, project: str = None) -> str:
    """Find the latest DNAnexus file ID matching a name pattern.

    Args:
        name_pattern: Filename or glob pattern.
        folder: Optional DNAnexus folder path.
        project: Optional DNAnexus project ID.

    Returns:
        File ID string (e.g., "file-xxxx").

    Raises:
        FileNotFoundError: If no file is found.
        subprocess.CalledProcessError: If dx command fails.
    """
    cmd = ['dx', 'find', 'data', '--name', name_pattern]
    if folder:
        cmd += ['--folder', folder]
    if project:
        cmd += ['--project', project]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result.check_returncode()
    lines = result.stdout.strip().splitlines()

    for line in reversed(lines):  # Start from most recent
        match = re.search(r'\((file-[a-zA-Z0-9]+)\)', line)
        if match:
            return match.group(1)

    raise FileNotFoundError(f"No file ID found in dx output matching pattern '{name_pattern}'")
    
    
def create_folder_if_not_exists(dx_folder_path):
    """Create a DNAnexus folder if it does not exist.

    Args:
        dx_folder_path: DNAnexus folder path to create.
    """
    command = f"dx mkdir -p {dx_folder_path}"  # Use `-p` to create parent folders as needed
    print(f"Running command: {command}")
    os.system(f"bash -c '{command}'")  # Run the command in bash
    

def download_files(files):
    """Download files from DNAnexus.

    Args:
        files: A (file_id, output_path) tuple or list of such tuples.

    Raises:
        ValueError: If input format is invalid.
    """
    # Normalize input
    if isinstance(files, tuple) and len(files) == 2:
        files = [files]
    elif isinstance(files, list):
        if not all(isinstance(f, (tuple, list)) and len(f) == 2 for f in files):
            raise ValueError("files must be a (file_id, output_path) tuple or a list of such tuples")
    else:
        raise ValueError("files must be a (file_id, output_path) tuple or a list of such tuples")

    # Download each file
    for file_id, output_path in files:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        cmd = f"dx download {file_id} -o {output_path}"
        print(f"Running: {cmd}")
        os.system(cmd)

        
def upload_files(files, dx_target="results"):
    """Upload files to DNAnexus.

    Args:
        files: File path, list of paths, or list of (path, dx_folder) tuples.
        dx_target: Default DNAnexus folder.

    Raises:
        ValueError: If input format is invalid.
    """
    # Normalize input
    if isinstance(files, str):
        files = [(files, dx_target)]
    elif isinstance(files, list):
        if all(isinstance(f, str) for f in files):
            files = [(f, dx_target) for f in files]
        elif all(isinstance(f, (tuple, list)) and len(f) == 2 for f in files):
            files = files
        else:
            raise ValueError("Invalid input. Use list of strings or list of (file, dx_folder) tuples.")

    # Upload each file
    for file_path, dx_target in files:
        if os.path.isfile(file_path):
            cmd = f"dx upload {file_path} --path {dx_target}/"
            os.system(cmd)
        else:
            print(f"File not found: {file_path}")

            
def upload_folders(folders, dx_target="results"):
    """Upload one or more folders to DNAnexus.

    Args:
        folders: Folder path, list of paths, or list of (path, dx_target) tuples.
        dx_target: Default DNAnexus folder.

    Raises:
        ValueError: If input format is invalid.
    """
    if isinstance(folders, str):
        folders = [(folders, dx_target)]
    elif isinstance(folders, list):
        # If list of strings, use default dx_target
        if all(isinstance(f, str) for f in folders):
            folders = [(f, dx_target) for f in folders]
        # If list of (local, dx) tuples, use as-is
        elif all(isinstance(f, (list, tuple)) and len(f) == 2 for f in folders):
            folders = folders
        else:
            raise ValueError("folders must be a string, list of strings, or list of (local, dx_target) tuples")

    for local_folder, target_path in folders:
        if os.path.isdir(local_folder):
            command = f"dx upload {local_folder} --recursive --path {target_path}/"
            os.system(command)
        else:
            print(f"Folder not found: {local_folder}")


# Load a file based on its file type (json, txt, csv, tsv, xlsx, etc.)
def load_file(file_path):
    """Load a file by extension into a Python object.

    Args:
        file_path: Path to the file.

    Returns:
        Loaded object: dict for JSON, str for TXT, DataFrame for CSV/TSV/XLSX,
        or None for unsupported types.

    Raises:
        Exception: If reading the file fails.
    """
    try:
        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".json":
            with open(file_path, 'r') as f:
                return json.load(f)

        elif file_extension == ".txt":
            with open(file_path, 'r') as f:
                return f.read().replace('\n', ',')

        elif file_extension == ".csv":
            return pd.read_csv(file_path)

        elif file_extension == ".tsv":
            return pd.read_csv(file_path, delimiter='\t')

        elif file_extension == ".xlsx":
            return pd.read_excel(file_path)

        else:
            logging.warning(f"Unsupported file type {file_extension} for {file_path}")
            return None

    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        raise
