import tensorflow
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, UpSampling2D, SpatialDropout2D, SpatialDropout3D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
from tensorflow.keras.layers import DepthwiseConv2D, Activation, BatchNormalization, PReLU,AveragePooling2D, Concatenate, concatenate, Layer, InputSpec
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras import layers
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.backend as K

import os
import numpy as np



#=================================================================================
# U-Net Simple 
#=================================================================================
def convSimple(prevlayer, filters, prefix, kernel=(3, 3), strides=(1, 1), useBN=True):
	conv = Conv2D(filters, kernel, padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_Conv")(prevlayer)
	if useBN:
		conv = BatchNormalization(name=prefix + "_BN")(conv)
	conv = Activation('relu', name=prefix + "_Activation")(conv)
	return conv

# input shape of form (256, 256, 1)
def getSimpleUNet(input_shape):
	img_input = Input((input_shape))
	x = convSimple(img_input, 32, "conv1_1")
	skip1 = x = convSimple(x, 32, "conv1_2")
	x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(x)

	x = convSimple(x, 64, "conv2_1")
	skip2 = x = convSimple(x, 64, "conv2_2")
	x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(x)

	x = convSimple(x, 128, "conv3_1")
	skip3 = x = convSimple(x, 128, "conv3_2")
	x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(x)

	x = convSimple(x, 256, "conv4_1")
	x = convSimple(x, 256, "conv4_2")
	x = convSimple(x, 256, "conv4_3")

	x = concatenate([UpSampling2D()(x), skip3], axis=3)
	x = convSimple(x, 128, "conv5_1")
	x = convSimple(x, 128, "conv5_2")

	x = concatenate([UpSampling2D()(x), skip2], axis=3)
	x = convSimple(x, 64, "conv6_1")
	x = convSimple(x, 64, "conv6_2")

	x = concatenate([UpSampling2D()(x), skip1], axis=3)
	x = convSimple(x, 32, "conv7_1")
	x = convSimple(x, 32, "conv7_2")

	#conv7 = SpatialDropout2D(0.2)(conv7)
	prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(x)
	model = Model(img_input, prediction)
	return model

#=================================================================================
# U-Net Full
#=================================================================================
# input shape of form (256, 256, 1)
def getFullUNet(input_shape):
	img_input = Input((input_shape))
	
	x = convSimple(img_input, 64, "Down_Block1_1")
	downBlock1Conv2 = x = convSimple(x, 64, "Down_Block1_2")
	x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="Down_Block1_Pool")(x)

	x = convSimple(x, 128, "Down_Block2_1")
	downBlock2Conv2 = x = convSimple(x, 128, "Down_Block2_2")
	x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="Down_Block2_Pool")(x)

	x = convSimple(x, 256, "Down_Block3_1")
	downBlock3Conv2 = x = convSimple(x, 256, "DownBlock3_2")
	x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="Down_Block3_Pool")(x)

	x = convSimple(x, 512, "Down_Block4_1")
	downBlock4Conv2 = x = convSimple(x, 512, "Down_Block4_2")
	x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="Down_Block4_Pool")(x)
	
	x = convSimple(x, 1024, "Across_1")
	x = convSimple(x, 1024, "Across_2")

	x = UpSampling2D(size=(2,2), name="Up_Block4_Up")(x)
	x = convSimple(x, 512, "Up_Block4_1", kernel=(2,2))
	x = concatenate([x, downBlock4Conv2])
	x = convSimple(x, 512, "Up_Block4_2")
	x = convSimple(x, 512, "Up_Block4_3")

	x = UpSampling2D(size=(2,2), name="Up_Block3_Up")(x)
	x = convSimple(x, 256, "Up_Block3_1", kernel=(2,2))
	x = concatenate([x, downBlock3Conv2])
	x = convSimple(x, 256, "Up_Block3_2")
	x = convSimple(x, 256, "Up_Block3_3")

	x = UpSampling2D(size=(2,2), name="Up_Block2_Up")(x)
	x = convSimple(x, 128, "Up_Block2_1", kernel=(2,2))
	x = concatenate([x, downBlock2Conv2])
	x = convSimple(x, 128, "Up_Block2_2")
	x = convSimple(x, 128, "Up_Block2_3")

	x = UpSampling2D(size=(2,2), name="Up_Block1_Up")(x)
	x = convSimple(x, 64, "Up_Block1_1", kernel=(2,2))
	x = concatenate([x, downBlock1Conv2])
	x = convSimple(x, 64, "Up_Block1_2")
	x = convSimple(x, 64, "Up_Block1_3")

	#x = SpatialDropout2D(0.2)(x)

	prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(x)
	model = Model(img_input, prediction)
	
	return model


#=================================================================================
# ResNet-50/U-Net
# Type of pooling does not matter as start decoding earlier
#=================================================================================
def getHybridResNetUNet(inputShape, weights=None):
	baseModel = tensorflow.keras.applications.ResNet50(input_shape=inputShape, weights=weights,include_top=False, pooling=None)

	skip1=baseModel.get_layer('conv1_relu').output #128x128x64
	skip2=baseModel.get_layer('conv2_block3_out').output #64x64x256
	skip3=baseModel.get_layer('conv3_block4_out').output #32x32x512
	skip4=baseModel.get_layer('conv4_block6_out').output #16x16x1024
	encoderFinal=baseModel.get_layer('conv5_block3_out').output #8x8x2048

	x=UpSampling2D(name="Up_B1_Up")(encoderFinal) #16x16x2048
	x = convSimple(x, 1024, "Up_B1_UpConv", kernel=(2,2)) #16x16x1024
	x=Concatenate()([x,skip4]) #16x16x2048
	x=convSimple(x,1024,"Up_B1_1") #16x16x1024
	x=convSimple(x,1024,"Up_B1_2") #16x16x1024

	x=UpSampling2D(name="Up_B2_Up")(x) #32x32x1024 
	x = convSimple(x, 512, "Up_B2_UpConv", kernel=(2,2)) #32x32x512
	x=Concatenate()([x,skip3]) #32x32x1024
	x=convSimple(x,512,"Up_B2_1") #32x32x512
	x=convSimple(x,512,"Up_B2_2") #32x32x512

	x=UpSampling2D()(x) #64x64x512
	x = convSimple(x, 256, "Up_B3_UpConv", kernel=(2,2)) #64x64x256
	x=Concatenate()([x,skip2]) #64x64x512
	x=convSimple(x,256,"Up_B3_1") #64x64x256
	x=convSimple(x,256,"Up_B3_2") # 64x64x256

	x=UpSampling2D()(x) #128x128x256
	x = convSimple(x, 128, "Up_B4_UpConv", kernel=(2,2)) #128x128x128
	x=Concatenate()([x,skip1]) #128x128x192
	x=convSimple(x,128,"Up_B4_1") #128x128x128
	x=convSimple(x,128,"Up_B4_2") #128x128x128

	x=UpSampling2D()(x) #256x256x128
	x=convSimple(x,64,"Up_B5_1") #256x256x64
	x=convSimple(x,64,"Up_B5_2") #256x256x64

	x = Conv2D(3, kernel_size= (3,3), strides=(1,1), padding= 'same')(x)  #256x256x1
	x=layers.Activation('sigmoid')(x)
	model=Model(inputs=baseModel.input,outputs=x)
	return model

#=================================================================================
# VGG16/UNet
# Type of pooling does not actually matter as we start decoding at an earlier step
# This model diverged (loss kept increasing) without batch normalization
#=================================================================================


def getHybridVGG16UNet(inputShape, weights=None):
	baseModel = tensorflow.keras.applications.VGG16(input_shape=inputShape, weights=weights,include_top=False, pooling=None)
	
	skip1=baseModel.get_layer('block1_conv2').output #256x256x64
	skip2=baseModel.get_layer('block2_conv2').output #128x128x128
	skip3=baseModel.get_layer('block3_conv3').output #64x64x256
	skip4=baseModel.get_layer('block4_conv3').output #32x32x512
	encoderFinal=baseModel.get_layer('block5_conv3').output #16x16x512

	x=UpSampling2D(name="Up_B1_Up")(encoderFinal)   #32x32x512
	x=Concatenate()([x,skip4]) #32x32x1024
	x=convSimple(x,512,"Up_B1_1", useBN=True) #32x32x512
	x=convSimple(x,512,"Up_B1_2", useBN=True) #32x32x512

	x=UpSampling2D(name="Up_B2_Up")(x) #64x64x256
	x = convSimple(x, 256, "Up_B2_UpConv", kernel=(2,2), useBN=True) #64x64x256
	x=Concatenate()([x,skip3]) #64x64x512
	x=convSimple(x,256,"Up_B2_1", useBN=True) #64x64x256
	x=convSimple(x,256,"Up_B2_2", useBN=True) #64x64x256

	x=UpSampling2D()(x) #128x128x256
	x = convSimple(x, 128, "Up_B3_UpConv", kernel=(2,2), useBN=True) #128x128x128
	x=Concatenate()([x,skip2]) #128x128x256
	x=convSimple(x,128,"Up_B3_1", useBN=True) #128x128x128
	x=convSimple(x,128,"Up_B3_2", useBN=True) #128x128x128

	x=UpSampling2D()(x) #256x256x128
	x = convSimple(x, 64, "Up_B4_UpConv", kernel=(2,2), useBN=True) #256x256x64
	x=Concatenate()([x,skip1]) #256x256x128
	x=convSimple(x,64,"Up_B4_1", useBN=True) #256x256x64
	x=convSimple(x,64,"Up_B4_2", useBN=True) #256x256x64

	x = Conv2D(3, kernel_size= (3,3), strides=(1,1), padding= 'same')(x)  #returns 256x256x1
	x=layers.Activation('sigmoid')(x)
	model=Model(inputs=baseModel.input,outputs=x)
	return model

#TODO: EPSIOLON

#=================================================================================
# DeepLab V3+ with XCeption
#=================================================================================
'''
# Based primary on: https://github.com/bonlime/keras-deeplab-v3-plus
# Compared to: https://github.com/mjDelta/deeplabv3plus-keras/blob/master/deeplabv3plus.py
Differences from primary:
* No manual padding. Would only be necessary if stride > 1 AND dilation > 1 (see documentation)
* This has good diagram of old xception to give idea of whne to use activation: https://www.mdpi.com/2227-7390/7/12/1170/htm
'''
class BilinearUpsampling(Layer):

	def __init__(self, upsampling=(2, 2), data_format=None, size=None, **kwargs):
		super(BilinearUpsampling, self).__init__(**kwargs)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
		self.input_spec = InputSpec(ndim=4)

	def compute_output_shape(self, input_shape):
		height = self.upsampling[0] * \
			input_shape[1] if input_shape[1] is not None else None
		width = self.upsampling[1] * \
			input_shape[2] if input_shape[2] is not None else None
		return (input_shape[0],
				height,
				width,
				input_shape[3])
	def call(self, inputs):
		return tensorflow.compat.v1.image.resize_bilinear(inputs, (int(inputs.shape[1]*self.upsampling[0]),
												   int(inputs.shape[2]*self.upsampling[1])))

	def get_config(self):
		config = {'upsampling': self.upsampling,
				  	'data_format': self.data_format}
		base_config = super(BilinearUpsampling, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
	
	@classmethod
	def from_config(cls, config):
		return super(BilinearUpsampling, cls).from_config(config)
'''
class BilinearUpsampling(Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
            input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
            input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])
    def call(self, inputs):
        return tensorflow.compat.v1.image.resize_bilinear(inputs, (int(inputs.shape[1]*self.upsampling[0]),
                                                   int(inputs.shape[2]*self.upsampling[1])))

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
'''
# depthActivation flag currently does nothing
def sepConv2D(x, numFilters, kernel, strides, dilation, nameBase, epsilon=1e-3,
				depthActivation=True):
	x = DepthwiseConv2D(kernel, strides=strides, dilation_rate=dilation,
						padding='same', use_bias=False,
						name="%s_DepthConv" % nameBase)(x)
	x = BatchNormalization(name="%s_1BN" % nameBase, epsilon=epsilon)(x)
	#x = Activation(tensorflow.nn.relu, name="%s_1Act" % (nameBase))(x)
	x = Conv2D(numFilters, (1,1), padding='same', use_bias=False, 
				name="%s_Conv" % nameBase)(x)
	return x

def xceptionBlock(x, numFilterList, nameBase, dilationRate,
					useConvShortCut, lastStride, startWithActivation=True):
	if useConvShortCut:
		shortCut = Conv2D(numFilterList[-1], kernel_size=1, 
							strides=(lastStride,lastStride), name="%s_Shortcut_Conv" % nameBase)(x)
		shortCut = BatchNormalization(name="%s_Shortcut_BN" % nameBase)(shortCut)
	else:
		shortCut = x

	if startWithActivation:
		x = Activation(tensorflow.nn.relu, name="%s_0Act" % nameBase)(x)
	# First
	x = sepConv2D(x, numFilterList[0], (3,3), strides=(1,1),
								dilation = (dilationRate, dilationRate),
								nameBase="%s_1SepConv" % nameBase)
	x = BatchNormalization(name="%s_1BN" % nameBase)(x)
	x = Activation(tensorflow.nn.relu, name="%s_1Act" % nameBase)(x)

	
	# Second
	x = sepConv2D(x, numFilterList[1], (3,3), strides=(1,1),
								dilation = (dilationRate, dilationRate),
								nameBase="%s_2SepConv" % nameBase)
	x = BatchNormalization(name="%s_2BN" % nameBase)(x)
	skip = x
	x = Activation(tensorflow.nn.relu, name="%s_2Act" % nameBase)(x)
	# Third
	x = sepConv2D(x, numFilterList[2], (3,3), strides=(lastStride,lastStride),
								dilation = (dilationRate, dilationRate),
								nameBase="%s_3SepConv" % nameBase)
	x = BatchNormalization(name="%s_3BN" % nameBase)(x)

	x = layers.add([x, shortCut])
	return x, skip

	

	#### depth_padding???

def getDeepLab(inputShape, outputStride=16, numClasses=1):

	if outputStride == 8:
		entryBlock3LastStride = 1
		middleBlockRate = 2
		exitBlockRateList = [2,4]
		atrousRateList = [12, 24, 36]
		encoderEndShape = tuple(int(element/outputStride) for element in inputShape)[0:2]
	elif outputStride == 16:
		entryBlock3LastStride = 2
		middleBlockRate = 1
		exitBlockRateList = [1,2]
		atrousRateList = [6, 12, 18]
		encoderEndShape = tuple(int(element/outputStride) for element in inputShape)[0:2]
	else:
		raise ValueError("Output stride must be one of 8 or 16")
	#-------------------------------------------------------
	# Entry flow
	#-------------------------------------------------------
	img_input = Input((inputShape))
	# Starting convolutions
	x = Conv2D(32, (3, 3), strides=(2, 2), name='Entry_1Conv', 
				use_bias=False, padding='same')(img_input)
	x = BatchNormalization(name='Entry_1BN')(x)
	x = Activation(tensorflow.nn.relu, name="Entry_1Act")(x)
	
	x = Conv2D(64, (3, 3), strides=(1, 1), name='Entry_2Conv',
				use_bias=False, padding='same')(x)
	x = BatchNormalization(name='Entry_2BN')(x)
	x = Activation(tensorflow.nn.relu, name="Entry_2Act")(x)
	x, _ = xceptionBlock(x, [128, 128, 128], 'Entry_Block1',
								dilationRate=1, useConvShortCut=True,
								lastStride=2, startWithActivation=False)
	x, skip = xceptionBlock(x, [256, 256, 256], 'Entry_Block2',
								dilationRate=1, useConvShortCut=True,
								lastStride=2, startWithActivation=True)
	x, _ = xceptionBlock(x, [728, 728, 728], 'Entry_Block3',
								dilationRate=1, useConvShortCut=True,
								lastStride=entryBlock3LastStride,
								startWithActivation=True)
	#-------------------------------------------------------
	# Middle flow
	#-------------------------------------------------------
	for index in range(16):
		x, _ = xceptionBlock(x, [728, 728, 728], 'Middle_Block%d' % (index),
								dilationRate=middleBlockRate, useConvShortCut=False,
								lastStride=1, startWithActivation=True)
	#-------------------------------------------------------
	# Exit flow
	#-------------------------------------------------------
	# LAST stride of 2 listed in paper figure but does not give right side
	# when considering output_stride
	x, _ = xceptionBlock(x, [728, 1024, 1024], 'Exit_Block1',
								dilationRate=exitBlockRateList[0], useConvShortCut=True,
								lastStride=1, startWithActivation=True)  
	numFilterList = [1536, 1536, 2048]
	for index in range(3):
		x = sepConv2D(x, numFilterList[index], (3,3), 
									strides=(1,1), dilation=(exitBlockRateList[1],exitBlockRateList[1]),
									epsilon=1e-5,
									nameBase="Exit_Block2_%dSepConv" % (index))

		x = BatchNormalization(name="Exit_Block2_%dBN" % index)(x)
		x = Activation(tensorflow.nn.relu, name="Exit_Block2_%dAct" % index)(x)

	#-------------------------------------------------------
	# Spatial pyramid pooling
	#-------------------------------------------------------     
	branch0 = Conv2D(256, (1,1), padding='same', use_bias=False, name="ASPP_Branch0_Conv")(x)
	branch0 = BatchNormalization(name="ASPP_Branch0_BN")(branch0)
	branch0 = Activation(tensorflow.nn.relu, name="ASPP_Branch0_Act")(branch0)

	branch1 = sepConv2D(x, 256, kernel=(3,3), strides=(1,1), 
						dilation=(atrousRateList[0],atrousRateList[0]),
						epsilon=1e-5,
						nameBase="ASPP_Branch1_SepConv")   
	branch1 = BatchNormalization(name="ASPP_Branch1_SepConv_2BN")(branch1)
	branch1 = Activation(tensorflow.nn.relu, name="ASPP_Branch1_SepConv_2Act")(branch1)

	branch2 = sepConv2D(x, 256, kernel=(3,3), strides=(1,1),
						dilation=(atrousRateList[1],atrousRateList[1]),
						epsilon=1e-5,
						nameBase="ASPP_Branch2_SepConv") 
	branch2 = BatchNormalization(name="ASPP_Branch2_SepConv_2BN")(branch2)
	branch2 = Activation(tensorflow.nn.relu, name="ASPP_Branch2_SepConv_2ACt")(branch2)

	branch3 = sepConv2D(x, 256, kernel=(3,3), strides=(1,1), 
						dilation=(atrousRateList[2],atrousRateList[2]),
						epsilon=1e-5,
						nameBase="ASPP_Branch3_SepConv") 
	branch3 = BatchNormalization(name="ASPP_Branch3_SepConv_2BN")(branch3)
	branch3 = Activation(tensorflow.nn.relu, name="ASPP_Branch3_SepConv_3Act")(branch3)

	branch4 = AveragePooling2D(name="ASPP_Branch4_AvgPool", pool_size=encoderEndShape)(x)
	branch4 = Conv2D(256,(1,1), name="ASPP_Branch4_Conv", padding="same", use_bias=False)(branch4)
	branch4 = BatchNormalization(name="ASPP_Branch4_BN")(branch4)
	branch4 = Activation(tensorflow.nn.relu, name="ASPP_Branch4_Act")(branch4)
	branch4=BilinearUpsampling(encoderEndShape, name="ASPP_Branch4_BilUp")(branch4)
	x=Concatenate()([branch0, branch1, branch2, branch3, branch4])

	x = Conv2D(256, (1,1), name="ASPP_PostBranch_Conv", padding="same", use_bias=False)(x)
	x = BatchNormalization(name="ASPP_PostBranch_BN")(x)
	x = Activation(tensorflow.nn.relu, name="ASPP_PostBranch_Act")(x)

	#model = Model(img_input, x)
	#return model

	#-------------------------------------------------------
	# Decoder
	#-------------------------------------------------------    
	# Upsample
	if outputStride == 16:
		upTuple = (4,4)
	elif outputStride == 8:
		upTuple = (2,2)
	else:
		raise ValueError("Invalid OS: %d" % (outputStride))
	x = BilinearUpsampling(upTuple, name="UP_BilUp")(x)
	# Flesh out the skip connection
	skip = Conv2D(48, (1,1), padding='same', use_bias=False, 
					name="Skip_Conv")(skip)
	skip = BatchNormalization(name="Skip_BN", epsilon=1e-5)(skip)
	skip = Activation(tensorflow.nn.relu, name="Skip_Act")(skip)
	# Merge the upsample with the skip connection
	x = Concatenate()([x, skip])

	x = sepConv2D(x, 256, kernel=(3,3), strides=(1,1), 
						dilation=(1, 1),
						epsilon=1e-5,
						nameBase="UP_SepConv1") 
	x = BatchNormalization(name="UP_SepConv1_2BN")(x)
	x = Activation(tensorflow.nn.relu, name="UP_SepConv1_2Act")(x)

	x = sepConv2D(x, 256, kernel=(3,3), strides=(1,1), 
						dilation=(1, 1),
						epsilon=1e-5,
						nameBase="UP_SepConv2") 
	x = BatchNormalization(name="UP_SepConv2_2BN")(x)
	x = Activation(tensorflow.nn.relu, name="UP_SepConv2_2Act")(x)

	x = Conv2D(numClasses, (1,1), padding="same")(x)
	x = BilinearUpsampling((4,4))(x)

	x = Activation(tensorflow.nn.sigmoid, name="Sigmoid_Act")(x)
	#x=layers.Activation('sigmoid')(x)
	
	model = Model(img_input, x)
	return model



if __name__ == "__main__":
	#model = getHybridResNetUNet((256,256,3))

	modelIndex = 4
	#------------------------------------------------------------
	# Simple U-Net
	#------------------------------------------------------------
	if modelIndex == 0:
		model = getSimpleUNet((256,256,1))
		modelNameBase = "UNet_Simple"
	#------------------------------------------------------------
	# Full UNet
	#------------------------------------------------------------
	elif modelIndex == 1:
		model = getFullUNet((256,256,1))
		modelNameBase = "UNet_Full"
	#------------------------------------------------------------
	# ResNet-50/U-Net
	#------------------------------------------------------------
	elif modelIndex == 2:
		model = getHybridResNetUNet((256,256,3))
		modelNameBase = "ResNet_UNet"
	#------------------------------------------------------------
	# VGG16/U-Net
	#------------------------------------------------------------
	elif modelIndex == 3:
		model = getHybridVGG16UNet((256,256,3))
		modelNameBase = "VGG16_UNet"
	#------------------------------------------------------------
	# DeepLab V3+ Model
	#------------------------------------------------------------
	elif modelIndex == 4:
		model = getDeepLab((256,256,1), outputStride=16)
		modelNameBase = "DeepLab_OS16"
	
	elif modelIndex == 22:
		model = tensorflow.keras.applications.xception.Xception(include_top=False, weights=None, input_shape=(256,256,3))
		modelNameBase = "Xception"
		




	print(model.summary())
	tensorflow.keras.utils.plot_model(model, to_file="/mnt/c/users/matth/Documents/Research/liver_paper/model_outputs/segment/%s.png" % (modelNameBase))
	stringList = []
	model.summary(line_length=210, print_fn=lambda x: stringList.append(x))
	summary = "\n".join(stringList)
	with open('/mnt/c/users/matth/Documents/Research/liver_paper/model_outputs/segment/%s.txt' % (modelNameBase), 'w') as outputFile:
		outputFile.write(summary)



