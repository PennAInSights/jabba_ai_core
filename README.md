# jabba_ai_core
Core functionality for jabba_ai based pipelines

## Build container
```apptainer build /path/to/jabba_ai_core.sif Apptainer```

## Process an image using the container
```sh /path/to/jabba_ai_core/tools/process_image_apptainer.sh /path/to/models_dir /path/to/jabba_ai_core.sif /path/to/myimage.nii.gz /path/to/output_dir```

This will generate the following outputs:
* /path/to/output_dir/myimage_liver.nii.gz
* /path/to/output_dir/myimage_spleen.nii.gz
* /path/to/output_dir/myimage_slab.nii.gz
* /path/to/output_dir/myimage_fats.nii.gz
* /path/to/output_dir/myimage_muscles.nii.gz
* /path/to/output_dir/myimage_imf.nii.gz

## Create virtual environment and install the package
```
python3.10 -m venv nalhutta
source nalhutta/bin/activate
nahutta/bin/pip3.10 install -r /path/to/jabba_ai_core/requirements.txt
nalutta/bin/pip3.10 install /path/to/jabba_ai_core
```

