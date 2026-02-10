# FAQ

## Do I need to upload Phenofhy manually to the TRE?

Yes. Phenofhy is currently a beta package and must be uploaded to the OFH TRE manually
(using the Airlock process). There is no automated installer yet, as OFH doesn't yet allow access to package repositories like PyPI.

## Where does `config.json` live?

Phenofhy expects `config.json` at `/mnt/project/helpers/config.json`.

## Do I need the `dx` CLI?

Yes. Phenofhy relies on DNAnexus tools for dataset access and metadata downloads. But note `dx` comes preinstalled on the OFH tre.

## What if my cohort key is missing?

`config.json` controls cohort keys under `COHORTS`. Add or update the key for your
study before running extraction.

## Where are metadata dictionaries stored?

Phenofhy downloads them into `./metadata` in your working directory when needed.
