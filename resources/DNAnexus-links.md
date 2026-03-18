# Learning Resources for the DNAnexus Trusted Research Environment

Curated guides and tutorials for researchers working within the UK Biobank Research Analysis Platform (UKB-RAP) or the Our Future Health Trusted Research Environment (OFH-TRE).

---

## Prerequisites

### Command-Line Fundamentals
- [Bash for Bioinformatics](https://laderast.github.io/bash_for_bioinformatics/) — A self-paced introduction to shell scripting for life scientists
- [Best Practices on the DNAnexus Platform](https://laderast.github.io/best_practices_dnanexus/) — Recommended workflows and conventions for reproducible cloud-based research

### The DNAnexus Command-Line Interface (dx toolkit)
- [Introduction to the CLI](https://academy.dnanexus.com/command_line_interface_cli/introduction_to_cli) — Official DNAnexus Academy guide
- [dx toolkit examples](https://github.com/dnanexus/OpenBio/tree/master/dx-toolkit) — Practical usage examples from the DNAnexus OpenBio repository
- [Getting started with dxdata](https://github.com/dnanexus/OpenBio/blob/master/dxdata/getting_started_with_dxdata.ipynb) — Notebook introduction to the dxdata Python library

### Our Future Health TRE Documentation
- [TRE help documentation](https://dnanexus.gitbook.io/ofh) — Platform features, functionality, and user guidance
- Our Future Health data documentation — Information on available datasets within the TRE
- Our Future Health researcher website — Researcher resources and application guidance

---

## Phenotypic Data Analysis in JupyterLab

### Loading and Exploring Data
- [Explore phenotypic tables](https://github.com/UK-Biobank/UKB-RAP-Notebooks-Access/blob/main/JupyterNotebook_Python/A101_Explore-phenotype-tables_Python.ipynb) — Navigating and inspecting phenotype table structure
- [Explore participant data](https://github.com/UK-Biobank/UKB-RAP-Notebooks-Access/blob/main/JupyterNotebook_Python/A102_Explore-participant-data_Python.ipynb) — Accessing individual-level participant records
- [Phenotypic data overview](https://github.com/dnanexus/OpenBio/blob/master/UKB_notebooks/ukb-rap-pheno-basic.ipynb) — Introductory notebook for phenotypic data on the RAP
- [OFH example TRE notebooks](https://github.com/ourfuturehealth/tre-example-notebooks/tree/main) — Worked examples specific to the Our Future Health TRE
- [Loading cohorts and analysing phenotypic data](https://github.com/dnanexus/OpenBio/blob/master/UKB_notebooks/loading_data_into_jupyterlab.ipynb) — End-to-end workflow from cohort definition to analysis
- [Export participant data](https://github.com/UK-Biobank/UKB-RAP-Notebooks-Access/blob/main/JupyterNotebook_Python/A103_Export-participant-data_Python.ipynb) — Extracting and saving processed outputs
- [Identifying participants with linked bulk files](https://github.com/UK-Biobank/UKB-RAP-Notebooks-Access/blob/main/JupyterNotebook_Python/A109_Find-participant-bulk-files.ipynb) — Locating participants with associated imaging or omics data

---

## Genomic Data Analysis in JupyterLab

### Getting Started with Hail
- [Hail tutorial on DNAnexus](https://github.com/dnanexus/OpenBio/tree/master/hail_tutorial) — Introduction to scalable genomic analysis using Hail on the RAP
- [GWAS tutorial (Hail documentation)](https://hail.is/docs/0.2/tutorials/01-genome-wide-association-study.html) — Step-by-step genome-wide association study using Hail

### Accessing and Linking Genomic Data
- [UKB-RAP genomics notebooks](https://github.com/UK-Biobank/UKB-RAP-Notebooks-Genomics) — Worked examples for accessing UK Biobank genetic data
- [Linking genotypic and phenotypic data](https://community.ukbiobank.ac.uk/hc/en-gb/community/posts/16019570597277-Query-of-the-week-7-About-linking-geno-and-pheno-data) — Community guide on merging genetic and phenotypic datasets

---

## Building Custom Applets and Pipelines

### Documentation
- [Building applets on DNAnexus](https://academy.dnanexus.com/buildingapplets) — Official Academy guide to packaging and deploying custom tools

### Video Tutorials
- [Running tools available on the UKB-RAP](https://www.youtube.com/watch?v=U8QZAGwnUm0) — Overview of the pre-installed tool library
- [Creating apps and bringing custom tools to the UKB-RAP](https://www.youtube.com/watch?v=A_iki_50Ig0) — Packaging your own software for use within the TRE
- [Introduction to WDL on the UKB-RAP](https://www.youtube.com/watch?v=2X3gbS_BHiA) — Workflow Description Language basics for pipeline development
- [Advanced WDL and Docker on the UKB-RAP](https://www.youtube.com/watch?v=vyzYChm0e1g) — Containerised workflows and advanced pipeline concepts

---

> **Note:** This resource list is maintained by Vincent Straub at the [Leverhulme Centre for Demographic Science](https://www.demography.ox.ac.uk/), University of Oxford. It is not affiliated with, endorsed by, or maintained by DNAnexus, UK Biobank, or Our Future Health.
