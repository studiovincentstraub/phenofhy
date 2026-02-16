# Environment and setup

Phenofhy is designed to run in the Our Future Health (OFH) trusted research environment (TRE) with DNAnexus tooling inside a `JupyterLab` notebook.

## Prerequisites

- Working knowledge of `Python` is requried and an understanding of how to launch and run analyses in the Our Future Health DNAnexus TRE using `JupyterLab`.
- For reference, an overview of resources for getting up to speed with DNAnexus TRE, the `dx toolkit`, and working on phenotypic data with `JupyterLab` is provided [here](https://github.com/studiovincentstraub/phenofhy/blob/main/resources/DNAnexus-links.md).

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

Phenofhy can in theory be used on similuated data outside the OFH TRE for local testing but it is intended to be run inside the TRE.