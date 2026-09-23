import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

WORKING_DIR = Path.cwd()
CONFIG_FILENAME = "config.json"
CONFIG_PATH = WORKING_DIR / CONFIG_FILENAME

PROJECT_DIR = WORKING_DIR
METADATA_DIR = WORKING_DIR / "metadata"
REMOTE_METADATA_DIR = "metadata"

REMOTE_CONFIG_DIR = "phenofhy"

def _ensure_remote_config_dir() -> None:
    subprocess.run(
        ["dx", "mkdir", "-p", REMOTE_CONFIG_DIR],
        check=True,
    )


def _upload_config(file_path: Path) -> str:
    output = _run(
        [
            "dx",
            "upload",
            str(file_path),
            "--path",
            f"{REMOTE_CONFIG_DIR}/",
            "--brief",
        ]
    )

    file_id_match = re.search(
        r"file-[A-Za-z0-9]+",
        output,
    )

    if not file_id_match:
        raise RuntimeError(
            f"Could not determine the DNAnexus file ID for "
            f"{file_path}. Command output was: {output}"
        )

    return file_id_match.group(0)

def _ensure_remote_metadata_dir() -> None:
    subprocess.run(
        ["dx", "mkdir", "-p", REMOTE_METADATA_DIR],
        check=True,
    )

def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Command failed with exit status {error.returncode}:\n"
            f"$ {' '.join(command)}\n"
            f"stdout:\n{error.stdout}\n"
            f"stderr:\n{error.stderr}"
        ) from error

    return result.stdout.strip()


def _get_project_id() -> str:
    project_id = os.environ.get("DX_PROJECT_CONTEXT_ID")

    if not project_id:
        raise RuntimeError(
            "DX_PROJECT_CONTEXT_ID is not set. "
            "Phenofhy init must be run inside a configured DNAnexus project."
        )

    return project_id


def _get_dataset_id(project_id: str) -> str:
    output = _run(
        [
            "dx",
            "find",
            "data",
            "--type",
            "Dataset",
            "--brief",
        ]
    )

    datasets = [
        line.strip()
        for line in output.splitlines()
        if line.strip()
    ]

    if not datasets:
        raise RuntimeError(
            "No DNAnexus Dataset was found in the current project."
        )

    dataset_id = datasets[0]

    if ":" in dataset_id:
        return dataset_id

    return f"{project_id}:{dataset_id}"


def _extract_metadata(dataset_id: str) -> dict[str, Path]:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "dx",
            "extract_dataset",
            dataset_id,
            "-ddd",
            "-o",
            str(METADATA_DIR),
        ]
    )

    metadata_files = {
        "CODINGS": next(
            METADATA_DIR.glob("*.codings.csv"),
            None,
        ),
        "DATA_DICT": next(
            METADATA_DIR.glob("*.data_dictionary.csv"),
            None,
        ),
        "ENTITY_DICT": next(
            METADATA_DIR.glob("*.entity_dictionary.csv"),
            None,
        ),
    }

    missing = [
        key
        for key, path in metadata_files.items()
        if path is None
    ]

    if missing:
        raise RuntimeError(
            "Metadata extraction did not produce: "
            + ", ".join(missing)
        )

    return metadata_files  # type: ignore[return-value]


def _upload_file(file_path: Path) -> str:
    output = _run(
        [
            "dx",
            "upload",
            str(file_path),
            "--path",
            f"{REMOTE_METADATA_DIR}/",
            "--brief",
        ]
    )

    file_id_match = re.search(
        r"file-[A-Za-z0-9]+",
        output,
    )

    if not file_id_match:
        raise RuntimeError(
            f"Could not determine the DNAnexus file ID for "
            f"{file_path}. Command output was: {output}"
        )

    return file_id_match.group(0)


def _build_config(
    project_id: str,
    dataset_id: str,
    metadata_files: dict[str, Path],
    metadata_ids: dict[str, str],
) -> dict[str, Any]:
    return {
        "PROJECT_ID": project_id,
        "PROJECT_DIR_PATH": "./",
        "BASE_PATHS": {
            "metadata": "metadata/",
            "phenofhy": "phenofhy/",
        },
        "COHORTS": {
            "FULL_SAMPLE": dataset_id,
        },
        "FILES": {
            "CODINGS": {
                "BASE": "metadata",
                "FILENAME": metadata_files["CODINGS"].name,
                "ID": metadata_ids["CODINGS"],
            },
            "DATA_DICT": {
                "BASE": "metadata",
                "FILENAME": metadata_files["DATA_DICT"].name,
                "ID": metadata_ids["DATA_DICT"],
            },
            "ENTITY_DICT": {
                "BASE": "metadata",
                "FILENAME": metadata_files["ENTITY_DICT"].name,
                "ID": metadata_ids["ENTITY_DICT"],
            },
        },
    }


def init(
    config_path: str | Path = CONFIG_PATH,
    dataset_id: str | None = None,
) -> Path:
    """Create, save, and upload a minimum working Phenofhy configuration."""

    print("Phenofhy initialization started.", flush=True)

    print("Finding project and dataset...", flush=True)
    project_id = _get_project_id()
    dataset_id = dataset_id or _get_dataset_id(project_id)
    print(f"Using dataset: {dataset_id}", flush=True)

    print("Preparing remote metadata directory...", flush=True)
    _ensure_remote_metadata_dir()

    print("Extracting metadata. This may take several minutes...", flush=True)
    metadata_files = _extract_metadata(dataset_id)
    print(
        "Metadata extracted: "
        + ", ".join(path.name for path in metadata_files.values()),
        flush=True,
    )

    print("Uploading metadata files...", flush=True)
    metadata_ids = {}

    for key, path in tqdm(
        metadata_files.items(),
        desc="Uploading metadata",
        unit="file",
    ):
        metadata_ids[key] = _upload_file(path)

    print("Building local config.json...", flush=True)
    config = _build_config(
        project_id=project_id,
        dataset_id=dataset_id,
        metadata_files=metadata_files,
        metadata_ids=metadata_ids,
    )

    output_path = Path(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print("Uploading config.json...", flush=True)
    _ensure_remote_config_dir()
    config_id = _upload_config(output_path)

    print(f"Phenofhy configuration created locally at: {output_path}")
    print(f"Phenofhy configuration uploaded to DNAnexus as: {config_id}")
    print("Phenofhy initialization complete.", flush=True)

    return output_path