# TRE Utilities

This tutorial shows common utilities in `phenofhy.utils` for working in the DNAnexus TRE. Use these helpers to locate datasets, move files in and out of projects, and load local metadata files during analysis setup. These functions complement the `dx toolkit` CLI rather than replace it, giving you quick shortcuts inside Python.

## Connect to a dataset

Use this when you want a quick handle to the current project dataset or cohort without manually copying IDs from the DNAnexus UI. It is a convenient wrapper around the usual `dx` workflow.

```python
from phenofhy.utils import get_dataset_id, connect_to_dataset

# Get the first dataset in the current project
full_id = get_dataset_id()

# Load the dataset (or a cohort if you have the record id)
dataset = connect_to_dataset()
```

## Find and download files

Use this when you need the most recent output that matches a pattern (for example, the latest CSV in `/results`) and want to download it locally. This complements `dx find` and `dx download` with a small Python helper.

```python
from phenofhy.utils import find_latest_dx_file_id, download_files

file_id = find_latest_dx_file_id("*.csv", folder="/results")

download_files((file_id, "outputs/latest_results.csv"))
```

## Upload files and folders

Use this to push outputs back into the project, especially when you want to script uploads from a notebook. It mirrors `dx upload` and `dx upload --recursive` but keeps the logic in Python.

```python
from phenofhy.utils import upload_files, upload_folders

upload_files(["outputs/latest_results.csv"], dx_target="/results")
upload_folders(["outputs/reports"], dx_target="/results")
```

## Load local files by extension

Use this to load small local files (JSON, CSV, TSV, XLSX) into Python without repeating boilerplate. It is useful for quick metadata checks before running a pipeline.

```python
from phenofhy.utils import load_file

cfg = load_file("config.json")
meta = load_file("metadata.tsv")
```

Notes:

- These helpers assume the DNAnexus CLI is available and authenticated.
- For dataset access, `DX_PROJECT_CONTEXT_ID` must be set in the environment.
