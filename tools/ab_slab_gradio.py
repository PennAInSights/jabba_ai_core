import gradio as gr
import os
from jabba_ai_core.ab_slab import AbSlab
from jabba_ai_core.core.jabba_custom_objects import get_jabba_custom_objects

import SimpleITK as sitk
import numpy as np
import logging

# Setup loggin
logging.basicConfig(level=logging.WARNING)
#logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger("ab_slab_gradio")
logger.setLevel(logging.DEBUG)

# Find model for detecting contrast enhanced CT images
model_name = os.environ['JABBA_SLAB_MODEL']
if not os.path.exists(model_name):
    raise ValueError(f"Model {model_name} does not exist")
    exit(1)
else:
    logger.debug(f"Model {model_name} exists")

# Preload the model
predictor = AbSlab(custom_objects=get_jabba_custom_objects())
#predictor.SetDebugOn()
predictor.LoadModel(model_name)

# Predict function for Gradio
def ab_slab(filename: str) -> list[int] | None:
    logger.debug("ab_slab()")

    if not os.path.isfile(filename):
        logging.error(f"File {filename} does not exist")
        return [None,None]
    img = sitk.ReadImage(filename, sitk.sitkFloat32)
    if img.GetDimension() != 3:
        logging.error("Input image is not 3D")
        return [None,None]     
    
    predictor.SetImage(img)
    predictor.Update()
    region = predictor.GetOutput()

    return( [ region['Index'][2], region['Size'][2] ] )

# Setup the Interface
title = "Jabba Abdonimal Slab Detection"
desc = "Determine the abdominal slab in a CT volume."
art="""# Description
Find the region of the abdominal slab in a 3D Nifti volume. The abdominal slab is defined as the region between the inferior aspect of the lung and inferior aspect of the L5 vertebrae.
\n\n# Usage
\n\n## Inputs
* filename = filename of a 3D Nifti volume
\n\n## Outputs
* output0 - Index of first slice in slab
* output1 - Number of slices in slab
\n\n# References
* <a href="https://academic.oup.com/jamia/article-abstract/28/6/1178/6133906">MacLean MT, Jehangir Q, Vujkovic M, Ko YA, Litt H, Borthakur A, Sagreiya H, Rosen M, Mankoff DA, Schnall MD, Shou H. Quantification of abdominal fat from computed tomography using deep learning and its association with electronic health records in an academic biobank. Journal of the American Medical Informatics Association. 2021 Jun 1;28(6):1178-87.</a>
"""

demo = gr.Interface(
    fn=ab_slab,
    inputs=["text"],
    examples=[["/data/ct_abdomen.nii.gz"]],
    outputs=[ gr.Number(), gr.Number() ],
    api_name="ab_slab",
    title=title,
    description=desc,
    article=art,
)
demo.launch()
