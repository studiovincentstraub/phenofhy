# Installation

Phenofhy 1.0 is distributed as a Python package and is designed for the Our
Future Health trusted research environment (TRE).

## Install in the OFH TRE

PyPI is not directly accessible from the OFH TRE. Our Future Health approves
packages for import into the TRE, after which approved packages can be
installed in a project. Phenofhy is currently going through that approval
process, so PyPI-based installation in the TRE should become available once
approval is complete.

Until then, use the Airlock workaround:

1. Obtain the Phenofhy source package through the Airlock process:
   https://dnanexus.gitbook.io/ofh/airlock/importing-files-into-a-restricted-project
2. Add the approved package files to a folder in your DNAnexus project.
3. At the start of each JupyterLab session, download the package into the
   notebook working directory:

   ```python
   !dx download "phenofhy:/phenofhy/v1/phenofhy/" -r
   ```

Replace the first `phenofhy` with your DNAnexus project name if it differs.
The remote folder should contain the package files directly, including
`__init__.py`, `config.py`, `load.py`, and the other modules. It should not
contain an additional nested `phenofhy/phenofhy/` directory.

Run the download command from the directory in which you want the local
`phenofhy/` package folder to be created. After downloading, verify the
package is importable:

```python
import phenofhy
print(phenofhy.__version__)
```

## Initialize a project

Run initialization once from a TRE JupyterLab notebook:

```python
import phenofhy

phenofhy.init()
```

Initialization discovers the current project and dataset, extracts and uploads
the three OFH metadata dictionaries, and uploads a generated `config.json` to
the project's `phenofhy/` folder. The local `metadata/` directory and config
are written to the notebook's working directory.

In later sessions, importing Phenofhy is sufficient. If the local config is
missing, Phenofhy downloads the project config automatically. Run `init()`
again only when the dataset or metadata configuration changes.

Initialization requires:

- `DX_PROJECT_CONTEXT_ID` set by the TRE;
- an authenticated `dx` toolkit;
- permission to read the selected dataset;
- permission to create or upload files in the project.

## Install from PyPI

PyPI is not directly accessible from the OFH TRE, but it is the recommended
installation method for local development and testing outside the TRE. This is
particularly useful for testing the simulation utilities in `simulate.py`,
which do not require DNAnexus access.

```bash
python -m pip install phenofhy
```

For development from a source checkout:

```bash
python -m pip install -e .
```

Phenofhy requires Python 3.10 or newer. See the [Simulating data locally](/tutorials/simulating-data-locally)
tutorial for a local testing example.
