import numpy as np
import SimpleITK as sitk

# SimpleITK DICOMOrient seems to output the opposite of is expected
def reorient_image(img, orient):
    map={"R":"L", "L":"R", "A":"P", "P":"A", "I":"S", "S":"I"}
    sorient = "".join([ map[x] for x in orient ])
    return(sitk.DICOMOrient(img, sorient))

def largest_connected_component(img):
    components = sitk.ConnectedComponent(img)
    sorted_components = sitk.RelabelComponent(components, sortByObjectSize=True)
    largest_component = sorted_components == 1
    return(largest_component)    

def largest_connected_component_per_label(img):
    labels = np.unique(sitk.GetArrayViewFromImage(img))
    labels = labels[labels!=0]

    outImg = img*0
    for l in labels:
        mask = sitk.BinaryThreshold(img, lowerThreshold=int(l), upperThreshold=int(l), insideValue=1, outsideValue=0)
        mask = largest_connected_component(mask)
        mask = mask*int(l)
        outImg = sitk.Add(outImg, mask)

    return(outImg)

def region_mask(img, size, index):
    mask = sitk.Image(img.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(img)
    mask[index[0]:(index[0]+size[0]-1), index[1]:(index[1]+size[1]-1), index[2]:(index[2]+size[2]-1)] = 1
    return(mask)

def resize_to_reference( img, reference, interpolation="Linear"):
    out_sp=reference.GetSpacing()
    out_sz=reference.GetSize()
    out_origin=reference.GetOrigin()
    out_dir=reference.GetDirection()

    resample = sitk.ResampleImageFilter()
    resample.SetSize( out_sz )
    resample.SetOutputSpacing( out_sp )
    resample.SetOutputOrigin( out_origin )
    resample.SetOutputDirection( out_dir )
    resample.SetTransform( sitk.Transform() )
    resample.SetDefaultPixelValue( 0 )

    if interpolation not in ["Linear","BSpline","NearestNeighbor"]:
        print("ERROR: unknown interpolation type:"+str(interpolation))
        return(None)

    if interpolation=="BSpline":
        resample.SetInterpolator( sitk.sitkBSpline )
    if interpolation=="NearestNeighbor":
        resample.SetInterpolator( sitk.sitkNearestNeighbor )

    return(resample.Execute(img))

def resize_image( img, outSize, outSpacing=None, outOrigin=None, interpolation="Linear" ):
    Dimension = img.GetDimension()
    inSize = img.GetSize()
    inSpacing = img.GetSpacing()
    inOrigin = img.GetOrigin()

    if outSpacing is None:
        outSpacing = [ y*float(inSize[x])/float(outSize[x]) for x,y in enumerate(inSpacing) ]

    if outOrigin is None:
        outOrigin = inOrigin

    resample = sitk.ResampleImageFilter()
    resample.SetSize( outSize )
    resample.SetOutputSpacing( outSpacing )
    resample.SetOutputOrigin( outOrigin )
    resample.SetOutputDirection( img.GetDirection() )
    resample.SetTransform( sitk.Transform() )
    resample.SetDefaultPixelValue( img.GetPixelIDValue() )

    if interpolation not in ["Linear","BSpline","NearestNeighbor"]:
        print("ERROR: unknown interpolation type:"+str(interpolation))
        return(None)

    if interpolation=="BSpline":
        resample.SetInterpolator( sitk.sitkBSpline )
    if interpolation=="NearestNeighbor":
        resample.SetInterpolator( sitk.sitkNearestNeighbor )

    return(resample.Execute(img))

def shift_scale_image(img, shift=0, scale=1, masked=False):
    out_img = sitk.ShiftScale(img, shift=shift, scale=scale)

    if masked:
        out_img = out_img * sitk.Cast(img>0, out_img.GetPixelID())
        
    return(out_img)


def window_level_image(img, window, level, pix_type=sitk.sitkUInt8):
    filter = sitk.IntensityWindowingImageFilter()
    filter.SetWindowMinimum(level-window/2)
    filter.SetWindowMaximum(level+window/2)
    filter.SetOutputMinimum(0)
    filter.SetOutputMaximum(255)

    return(sitk.Cast(filter.Execute(img), pix_type))

# Zero out any voxels with multiple labels
# FIXME: Better to do fusion via predicted weights
def label_fusion(masks, labels):

    # check for multi masks
    if len(masks)==1:
        return(masks[0])
    
    count = masks[0] + masks[1]
    if len(masks)>2:
        for i in range(len(masks)-2):
            count = count + masks[i+2]

    oneLabel = sitk.BinaryThreshold(count, lowerThreshold=0, upperThreshold=1, insideValue=1, outsideValue=0)
    seg = sitk.BinaryThreshold(masks[0], lowerThreshold=1, upperThreshold=1, insideValue=labels[0], outsideValue=labels[0])
    for i in range(len(masks)-1):
        iseg = oneLabel * masks[i+1]
        iseg = sitk.BinaryThreshold(iseg, lowerThreshold=1, upperThreshold=1, insideValue=labels[i+1])
        seg = seg + iseg

    return(seg)

# later masks overwrite earlier masks
def label_merge(masks, labels):
    arr = sitk.GetArrayFromImage(masks[0])
    arr[arr>0]=labels[0]

    for i in range(len(masks)-1):
        iarr = sitk.GetArrayViewFromImage(masks[i+1])
        arr[iarr>0]=labels[i+1]

    seg = sitk.GetImageFromArray(arr)
    seg.CopyInformation(masks[0])
    return(seg)


def label_overlay(labels, feature, opacity=0.5, colormap=None, backgroundValue=0):    

    filter = sitk.LabelOverlayImageFilter()
    filter.SetOpacity(opacity)
    filter.SetBackgroundValue(backgroundValue)
    if colormap is not None:
        filter.SetColormap(colormap)
    return(filter.Execute(feature, labels))






