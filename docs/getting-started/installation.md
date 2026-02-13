# Installation

Phenofhy is currently a beta package designed to run inside the Our Future Health TRE.
There is no automated installer yet; you must upload the package manually.

## Step 1: Download the package

Download a zip of the Phenofhy package from GitHub:

https://github.com/studiovincentstraub/phenofhy/tree/main/beta/phenofhy

Tip: download the entire repository as a zip and just upload `phenofhy/` folder
intact when you upload it to the TRE (do not nest `phenofhy` in `phenofhy/beta/phenofhy`).

## Step 2: Upload to the TRE (Airlock)

Phenofhy must be transferred into the TRE using the Airlock process.
The Airlock team will guide you through each stage as the request is processed.

Summary of the modified ingress route:

1. Upload the package to an unrestricted project in the OFH TRE.
2. Submit an import request and follow Airlock instructions:
   https://dnanexus.gitbook.io/ofh/airlock/importing-files-into-a-restricted-project
3. Airlock performs security checks and transfers files into a restricted project.
4. Copy the approved files into your own restricted project and confirm completion.

## Step 3: Set `config.json`

Phenofhy expects a `config.json` file in the project helpers directory:

- Location: `/mnt/project/helpers/config.json`
- Purpose: identifies dataset dictionaries, phenotype list files, and base paths.

A template is included in the package under `phenofhy/beta/phenofhy/config.json`.
Copy it into the TRE helpers directory and update the IDs for your study:

- `PROJECT_ID` and `PROJECT_ID_LONG`
- `COHORTS` entries (for example, `FULL_SAMPLE` and `TEST_COHORT`). These can be created using the cohort browser and saving cohort dashboard views; for instructions, see: https://documentation.dnanexus.com/user/cohort-browser/defining-cohorts
- `FILES` entries for the OFH [coding and data dictionary files](https://research.ourfuturehealth.org.uk/data-and-cohort/), should you have already created and saved these in an OFH project. If not, Phenofhy will automatically create them for you using `phenofhy.load.metadata()` and store them in a `metadata/` directory. 

You will usually edit this file from a JupyterLab instance in the TRE, but you can
also update it locally before uploading. Expect to refresh file IDs over time as
datasets and dictionary files are updated.

## After transfer

Once the package and `config.json` are in place, you can follow the Quickstart.
