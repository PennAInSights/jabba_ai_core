import gradio as gr
import os
from jabba_ai_core.ab_contrast import AbContrast
from jabba_ai_core.core.jabba_custom_objects import get_jabba_custom_objects
import SimpleITK as sitk
import numpy as np
import logging

# Setup loggin
logging.basicConfig(level=logging.WARNING)
#logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger("ab_contrast_gradio")
logger.setLevel(logging.DEBUG)

# Find model for detecting contrast enhanced CT images
ab_contrast_model_name = os.environ['JABBA_CONTRAST_MODEL']
if not os.path.exists(ab_contrast_model_name):
    raise ValueError(f"Model {ab_contrast_model_name} does not exist")
    exit(1)
else:
    logger.debug(f"Model {ab_contrast_model_name} exists")

# Preload the model
predictor = AbContrast(custom_objects=get_jabba_custom_objects())
#predictor.SetDebugOn()
predictor.LoadModel(ab_contrast_model_name)

# Predict function for Gradio
def ab_contrast(filename: str, return_type: str) -> float | int | None:
    logger.debug("ab_contrast()")

    if not os.path.isfile(filename):
        logging.error(f"File {filename} does not exist")
        return None
    img = sitk.ReadImage(filename, sitk.sitkFloat32)
    if img.GetDimension() != 3:
        logging.error("Input image is not 3D")
        return None     
    
    predictor.SetImage(img)
    predictor.Update()
    chances = predictor.GetOutput()

    if return_type == "mean":
        return( float(np.mean(chances)) )
    elif return_type == "bool":
        return( int(np.mean(chances) > 0.5) )


# Setup the Interface
title = "Jabba Contrast Detection"
desc = "Determine the likelihood of contrast enhancement in a CT volume of the abdomen"
art="""# Description
This model predicts the likelihood of contrast enhancement in a CT volume of the abdomen. The model is based on a deep learning model trained on a large dataset of CT volumes. The model outputs a probability of contrast enhancement, which can be used to determine if a volume is likely to contain contrast. The model is designed to be used in a clinical setting to help radiologists quickly identify volumes that may contain contrast.
\n\n# Inputs
* filename = filename of a 3D Nifti volume
* return_type = mean | bool
  * mean = the mean probability of contrast enhancement in the volume
  * bool = 1 if the probability is greater than 0.5, 0 otherwise
# References
* Citation goes here"""

demo = gr.Interface(
    fn=ab_contrast,
    inputs=["text", "text"],
    examples=[["/data/ct_abdomen.nii.gz", "mean"], ["/data/ct_abdomen.nii.gz", "bool"]],
    outputs=["number"],
    api_name="ab_contrast",
    title=title,
    description=desc,
    article=art,
)
demo.launch()
