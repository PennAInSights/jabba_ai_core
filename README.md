# jabba_ai_core
Core functionality for jabba_ai based pipelines

## Build container
apptainer build /path/to/jabba_ai_core.sif Apptainer

## Process an image using the container
sh /path/to/jabba_ai_core/tools/process_image_apptainer.sh /path/to/models_dir /path/to/jabba_ai_core.sif /path/to/image.nii.gz /path/to/output_dir
