# Utils

DNAnexus helpers and file utilities.

## get_dataset_id()

```python
phenofhy.utils.get_dataset_id(project=None, full=True)
```

Return the DNAnexus dataset identifier.

**Parameters**

&nbsp;&nbsp;**project**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;DNAnexus project ID. Defaults to DX_PROJECT_CONTEXT_ID.<br>
&nbsp;&nbsp;**full**: ***bool***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If True, return "{project}:{record_id}". If False, return project ID only.<br>

**Returns**

&nbsp;&nbsp;**out**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dataset identifier string.<br>

**Raises**

&nbsp;&nbsp;**RuntimeError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If project is not available.

## connect_to_dataset()

```python
phenofhy.utils.connect_to_dataset(cohort_record_id="")
```

Connect to a DNAnexus dataset or cohort.

**Parameters**

&nbsp;&nbsp;**cohort_record_id**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional cohort record ID. If empty, loads the first dataset.<br>

**Returns**

&nbsp;&nbsp;**out**: ***dxdata.Dataset | dxdata.Cohort***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Loaded DNAnexus dataset or cohort.

**Raises**

&nbsp;&nbsp;**RuntimeError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If DX_PROJECT_CONTEXT_ID is not set.

## find_latest_dx_file_id()

```python
phenofhy.utils.find_latest_dx_file_id(name_pattern, folder=None, project=None)
```

Find the latest DNAnexus file ID matching a name pattern.

**Parameters**

&nbsp;&nbsp;**name_pattern**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Filename or glob pattern.<br>
&nbsp;&nbsp;**folder**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional DNAnexus folder path.<br>
&nbsp;&nbsp;**project**: ***str | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Optional DNAnexus project ID.<br>

**Returns**

&nbsp;&nbsp;**out**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;File ID string (e.g., "file-xxxx").

**Raises**

&nbsp;&nbsp;**FileNotFoundError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If no file is found.<br>
&nbsp;&nbsp;**subprocess.CalledProcessError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If dx command fails.

## download_files()

```python
phenofhy.utils.download_files(files)
```

Download files from DNAnexus.

**Parameters**

&nbsp;&nbsp;**files**: ***tuple | list***<br>
&nbsp;&nbsp;&nbsp;&nbsp;A (file_id, output_path) tuple or list of such tuples.<br>

**Raises**

&nbsp;&nbsp;**ValueError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If input format is invalid.

## upload_files()

```python
phenofhy.utils.upload_files(files, dx_target="results")
```

Upload files to DNAnexus.

**Parameters**

&nbsp;&nbsp;**files**: ***str | list***<br>
&nbsp;&nbsp;&nbsp;&nbsp;File path, list of paths, or list of (path, dx_folder) tuples.<br>
&nbsp;&nbsp;**dx_target**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Default DNAnexus folder.<br>

**Raises**

&nbsp;&nbsp;**ValueError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If input format is invalid.

## upload_folders()

```python
phenofhy.utils.upload_folders(folders, dx_target="results")
```

Upload one or more folders to DNAnexus.

**Parameters**

&nbsp;&nbsp;**folders**: ***str | list***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Folder path, list of paths, or list of (path, dx_target) tuples.<br>
&nbsp;&nbsp;**dx_target**: ***str***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Default DNAnexus folder.<br>

**Raises**

&nbsp;&nbsp;**ValueError**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If input format is invalid.

## load_file()

```python
phenofhy.utils.load_file(file_path)
```

Load a file by extension into a Python object.

**Parameters**

&nbsp;&nbsp;**file_path**: ***str | pathlib.Path***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Path to the file.<br>

**Returns**

&nbsp;&nbsp;**out**: ***dict | str | pandas.DataFrame | None***<br>
&nbsp;&nbsp;&nbsp;&nbsp;Loaded object based on file extension, or None if unsupported.

**Raises**

&nbsp;&nbsp;**Exception**: ***Exception***<br>
&nbsp;&nbsp;&nbsp;&nbsp;If reading the file fails.

**Example**
```python
from phenofhy import utils

dataset_id = utils.get_dataset_id()
```
