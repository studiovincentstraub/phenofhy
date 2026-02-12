# phenofhy (beta)

Python package for phenotype analysis in the Our Future Health (OFH) trusted
research environment (TRE). phenofhy is designed to make extraction, processing,
and reporting of OFH phenotype data quick and repeatable inside DNAnexus.

![cover image](logo/cover.png)

## Why phenofhy

- Purpose-built for the OFH TRE and DNAnexus tooling.
- Easily extract and preprocess phenotype data in a few lines of intuitive code.
- Quick summaries and basic phenotype profile reporting to validate data early.

## Target users

- Researchers and students wanting to get started with analysing OFH phenotypes.
- Teams working inside the OFH TRE who need a repeatable preprocessing workflow.
- Analysts creating quick QA summaries and phenotype profile reports before GWAS.

## Installation (TRE)

phenofhy is currently a beta package for the OFH TRE. There is no automated
installer yet.

1. Download a zip of the repository from GitHub:
   https://github.com/studiovincentstraub/phenofhy
2. Upload the `phenofhy/` folder into the TRE using the Airlock process.
   Guidance: https://dnanexus.gitbook.io/ofh/airlock/importing-files-into-a-restricted-project
3. Copy `beta/phenofhy/config.json` into `/mnt/project/helpers/config.json` and
   update the IDs for your study (project IDs, cohorts, codings, dictionaries).

Tip: when you upload the package, avoid nesting `phenofhy` inside another
`phenofhy` folder. The TRE should contain a single `phenofhy/` directory that
includes the beta modules, which can be optionally nested inside a `applets/` folder.

## Environment

phenofhy is designed to run inside the OFH TRE with DNAnexus tooling and
JupyterLab. It can be used on simulated data outside the TRE for local testing,
but the main workflows assume access to OFH datasets and the `dx` toolkit.

## Documentation

Explore the full phenofhy documentation here: https://studiovincentstraub.github.io/phenofhy/
Even though the documentation website is still under construction, you can
already find useful information there.

Where to start on the documentation website?

- New to phenofhy or OFH phenotype analysis? Begin with "About" and then the
	"Quickstart" for a smooth introduction.
- Got your own data? After "About" and "Quickstart", you are ready to dive in
	and start analyzing.
- Looking for more? Check out the example workflows and API reference to deepen
	your understanding.

## Contributing

phenofhy is an internal beta and is evolving quickly. If you find a bug or want
to suggest an improvement, open an issue or start a discussion in the
repository. 

