import gradio as gr
import os
from jabba_ai_core.ab_spleen import AbSpleen
from jabba_ai_core.core.jabba_custom_objects import get_jabba_custom_objects
import SimpleITK as sitk
import numpy as np
import logging
import tensorflow as tf

# Setup loggin
logging.basicConfig(level=logging.WARNING)
#logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger("ab_spleen_gradio")
logger.setLevel(logging.DEBUG)

# Find model for detecting contrast enhanced CT images
ab_spleen_model_name = os.environ['JABBA_SPLEEN_MODEL']
if not os.path.exists(ab_spleen_model_name):
    raise ValueError(f"Model {ab_spleen_model_name} does not exist")
    exit(1)
else:
    logger.debug(f"Model {ab_spleen_model_name} exists")

# Preload the model
AbSpleen.model =  tf.keras.models.load_model(ab_spleen_model_name, custom_objects=get_jabba_custom_objects())
predictor = AbSpleen()
predictor.SetDebugOn()
predictor.SetModel(AbSpleen.model)

# Predict function for Gradio
def ab_liver(filename: str, output_filename: str) -> float | None:
    logger.debug("ab_spleen()")

    if not os.path.isfile(filename):
        logging.error(f"File {filename} does not exist")
        return None
    img = sitk.ReadImage(filename, sitk.sitkFloat32)
    if img.GetDimension() != 3:
        logging.error("Input image is not 3D")
        return None     
    
    predictor.SetImage(img)

    try: 
        predictor.Update()
        seg_img = predictor.GetOutput()
        sitk.WriteImage(seg_img, output_filename)
        vol=np.sum(sitk.GetArrayFromImage(seg_img))*np.prod(seg_img.GetSpacing())
        return(vol)
    except:
        return(None)



# Setup the Interface
title = "Jabba Spleen Segmentation"
desc = "Segment the spleen in a CT volume of the abdomen"
art="""# Description
This model segments the spleen in a non-contrast enhanced abdominal CT volume
\n\n# Inputs
* filename = filename of a 3D Nifti volume
* output_filename = filename of the output volume
* return_type = mean | bool
  * mean = the mean probability of contrast enhancement in the volume
  * bool = 1 if the probability is greater than 0.5, 0 otherwise
# References
* Citation goes here"""

demo = gr.Interface(
    fn=ab_liver,
    inputs=["text", "text"],
    examples=[["/data/ct_abdomen.nii.gz", "/data/ct_abdomen_spleen.nii.gz"]],
    outputs=["number"],
    api_name="ab_liver",
    title=title,
    description=desc,
    article=art,
)
demo.launch()
