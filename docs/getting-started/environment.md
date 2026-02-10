# Environment and setup

Phenofhy is designed to run in the Our Future Health TRE with DNAnexus tooling inside a JupyterLab notebook.

## Requirements

- All you need is an active OFH TRE project on the DNAnexus platform and working knowledge of how to use JupyterLab (see [Introduction to Jupyterlab](https://dnanexus.gitbook.io/ofh/jupyterlab/introduction-to-jupyterlab))
- `DX_PROJECT_CONTEXT_ID` set (already configured automatically in the TRE).
- It is recommened you configure a `config.json` file in `/mnt/project/helpers` with file IDs and base paths (see Installation).

## Metadata files

Phenofhy can automatically create the [metadata files](https://ourfuturehealth.gitbook.io/our-future-health/data/participant-data#what-metadata-is-available-to-help-document-the-data-release) (codings, data_dictionary, entity_dictionary) that come with OFH and which can also be created with `dx` (see [documentation provided by UK Biobank](https://dnanexus.gitbook.io/uk-biobank-rap/working-on-the-research-analysis-platform/accessing-data/accessing-phenotypic-data#programatically)).

The helper `load.metadata()` downloads them into `./metadata` if missing.

```python
from phenofhy import load

meta = load.metadata()
# meta["codings"], meta["data_dictionary"], meta["entity_dictionary"]
```

## TRE-only usage

Phenofhy is intended to run inside the OFH TRE. It can in theory be used on similuated data outside the TRE for local testing. 