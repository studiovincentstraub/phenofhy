# FAQ

## How do I install Phenofhy?

Install it from PyPI when the TRE permits package installation:

```bash
python -m pip install phenofhy
```

If the TRE cannot access PyPI, transfer the wheel through the Airlock process.

## Where does `config.json` live?

Run `phenofhy.init()` once. It creates a local config in the notebook working
directory and uploads the project copy to `phenofhy/config.json`.

## Do I need the `dx` CLI?

Yes. Phenofhy relies on DNAnexus tools for dataset access and metadata downloads. But note `dx` comes preinstalled on the OFH tre.

## What if my cohort key is missing?

The generated `config.json` controls cohort keys under `COHORTS`. The default
dataset is available as `COHORTS["FULL_SAMPLE"]`; pass another cohort key when
using a custom configuration.

## Where are metadata dictionaries stored?

Phenofhy downloads them into `./metadata` in your working directory when needed.
