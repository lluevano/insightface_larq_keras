import tensorflow as tf

import larq as lq
from tensorflow.keras.layers import (Conv2D, Add, Reshape, Multiply, PReLU, BatchNormalization, Input, DepthwiseConv2D)
from larq.layers import (QuantConv2D, QuantSeparableConv2D, QuantLocallyConnected2D, QuantDepthwiseConv2D)
from larq.quantizers import (SteSign, ApproxSign, MagnitudeAwareSign, DoReFa)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def conv_block(inputs, filters, kernel_size, strides, padding,groups=1):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    #channel_axis = -1
    Z = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False,groups=groups)(inputs)
    Z = BatchNormalization(axis=channel_axis)(Z)
    A = PReLU(shared_axes=[1, 2])(Z)
    return A

def qconv_block(inputs, filters, kernel_size, strides, padding="valid", activation=True, groups=1):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    kwargs = dict(input_quantizer=DoReFa(k_bit=1, mode="weights"),
                  kernel_quantizer=DoReFa(k_bit=1, mode="weights"),
                  kernel_constraint="weight_clip")
    #kwargs = dict(input_quantizer="ste_sign",
    #              kernel_quantizer="ste_sign",
    #              kernel_constraint="weight_clip")
    #channel_axis = -1
    Z = QuantConv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, groups=groups, **kwargs)(inputs)
    Z = BatchNormalization(axis=channel_axis)(Z)
    if activation:
        A = PReLU(shared_axes=[1, 2])(Z)
    else:
        A = Z
    return A

def my_reductionv2(inputs, filters):
    #input 112x112
    nn = qconv_block(inputs,filters,kernel_size=5,strides=3,padding="same") #38x38
    nn = conv_block(nn,filters,kernel_size=3,strides=3,padding="valid") #12x12
    nn = conv_block(nn,filters,kernel_size=3,strides=2,padding="valid") #5x5
    nn = conv_block(nn,filters,kernel_size=5,strides=5,padding="valid") #1x1
    return nn
    
def my_qreduction(inputs, filters): #turns 112x112 block to 1x1
    #nn = Conv2D(filters, kernel_size=5, strides=(2, 2))(inputs)  # 54x54
    nn = qconv_block(inputs, filters, kernel_size=5, strides=2, padding="valid") #14 x 14
    nn = qconv_block(nn, filters, kernel_size=3, strides=3, padding="valid") #4x4 
    nn = qconv_block(nn, filters, kernel_size=4, strides=1, padding="valid")
    #nn = qconv_block(nn, filters, kernel_size=1, strides=5, padding="valid")
    return nn

def my_qreductionv2(inputs, filters): #turns 112x112 block to 1x1
    nn = qconv_block(inputs,filters,kernel_size=5,strides=2,padding="valid",groups=4)
    nn = qconv_block(nn,filters,kernel_size=3,strides=3,padding="valid", groups=4)
    nn = qconv_block(nn,filters,kernel_size=1,strides=4,padding="valid",groups=2)
    nn = qconv_block(nn,filters,kernel_size=1,strides=5,padding="valid")
    
    return nn

def my_reduction(inputs, filters): #turns 112x112 block to 1x1
    nn = conv_block(inputs,filters,kernel_size=5,strides=2,padding="valid",groups=4)
    nn = conv_block(nn,filters,kernel_size=3,strides=3,padding="valid", groups=4)
    nn = conv_block(nn,filters,kernel_size=1,strides=4,padding="valid",groups=2)
    nn = conv_block(nn,filters,kernel_size=1,strides=5,padding="valid")
    
    return nn

def my_se_block(inputs, reduction=16):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1 #probs
    #channel_axis = -1
    filters = inputs.shape[channel_axis]
    #nn = GlobalAveragePooling2D()(inputs)
    #REPLACING GAP
    nn = my_qreductionv2(inputs, filters)
    #print(nn.shape)
    #CONTINUE NORMAL SE
    nn = Reshape((1, 1, filters))(nn)
    nn = Conv2D(filters // reduction, kernel_size=1)(nn)
    nn = PReLU(shared_axes=[1, 2])(nn)
    nn = Conv2D(filters, kernel_size=1, activation="sigmoid")(nn)
    nn = Multiply()([inputs, nn])
    return nn


def head_setting(inputs, channels=64):
    #Expand to 64 channels
    nn = conv_block(inputs,channels,(1,1),strides=(1,1),padding="same")
    nn = my_se_block(nn)
    return nn

def global_feature_extraction(inputs, channels=64):
    kwargs = dict(input_quantizer=DoReFa(k_bit=1, mode="weights"),
                  kernel_quantizer=DoReFa(k_bit=1, mode="weights"),
                  kernel_constraint="weight_clip")
    #kwargs = dict(input_quantizer="ste_sign",
    #            kernel_quantizer="ste_sign",
    #            kernel_constraint="weight_clip")
    sep_kwargs = dict(input_quantizer=DoReFa(k_bit=1, mode="weights"),
                      depthwise_quantizer=DoReFa(k_bit=1, mode="weights"),
                      depthwise_constraint="weight_clip",
                      pointwise_quantizer=DoReFa(k_bit=1, mode="weights"),
                      pointwise_constraint="weight_clip")
    depthwise_kwargs = dict(input_quantizer=DoReFa(k_bit=1, mode="weights"),
                            depthwise_quantizer=DoReFa(k_bit=1,mode="weights"),
                            depthwise_constraint="weight_clip",
                            bias_constraint="weight_clip")
    #sep_kwargs = dict(input_quantizer="ste_sign",
    #                  depthwise_quantizer="ste_sign",
    #                  depthwise_constraint="weight_clip",
    #                  pointwise_quantizer="ste_sign",
    #                  pointwise_constraint="weight_clip")

    skip1 = QuantConv2D(channels,(1,1),strides=(1,1),padding="same", **kwargs)(inputs)
    skip1 = PReLU(shared_axes=[1, 2])(skip1)
    sep1 = QuantDepthwiseConv2D((1,1),strides=(1,1),padding="same", **depthwise_kwargs)(skip1)
    #sep1 = QuantSeparableConv2D(channels,(1,1),strides=(1,1),padding="same", **sep_kwargs)(skip1)
    join1 = Add()([inputs,sep1])
    join1 = PReLU(shared_axes=[1,2])(join1)
    qnn = QuantConv2D(channels,(1,1),strides=(1,1),padding="same", **kwargs)(join1)
    join2 = Add()([qnn,skip1])
    join2 = PReLU(shared_axes=[1,2])(join2)
    sep2 = QuantDepthwiseConv2D((1,1),strides=(1,1),padding="same", **depthwise_kwargs)(join2)
    #sep2 = QuantSeparableConv2D(channels,(1,1),strides=(1,1),padding="same", **sep_kwargs)(join2)
    join3 = Add()([join2,sep2,inputs])
    join3 = PReLU(shared_axes=[1,2])(join3)
    return join3

def local_feature_extraction(inputs, channels=64):
    kwargs = dict(input_quantizer=DoReFa(k_bit=1, mode="weights"),
                  kernel_quantizer=DoReFa(k_bit=1, mode="weights"),
                  kernel_constraint="weight_clip")
                  
    #kwargs = dict(input_quantizer="ste_sign",
    #            kernel_quantizer="ste_sign",
    #            kernel_constraint="weight_clip")

    sep_kwargs = dict(input_quantizer=DoReFa(k_bit=1, mode="weights"),
                      depthwise_quantizer=DoReFa(k_bit=1, mode="weights"),
                      depthwise_constraint="weight_clip",
                      pointwise_quantizer=DoReFa(k_bit=1, mode="weights"),
                      pointwise_constraint="weight_clip")
                      
    #skip1 = QuantLocallyConnected2D(channels,(1,1),strides=(1,1),padding="valid", implementation=1, **kwargs)(inputs) #downsize to 56x56
    #skip1 = PReLU(shared_axes=[1,2])(skip1)
    skip1 = qconv_block(inputs, channels, kernel_size=3, strides=2, padding="same")
    #print(skip1.shape)
    #sep1 = QuantDepthwiseConv2D((1,1),strides=(1,1),padding="same")(skip1)
    sep1 = QuantSeparableConv2D(channels,(1,1),strides=(1,1),padding="same", **sep_kwargs)(skip1)
    sep1 = PReLU(shared_axes=[1,2])(sep1)
    #local2 = QuantLocallyConnected2D(channels,(1,1),strides=(1,1),padding="valid", implementation=1, **kwargs)(sep1)
    local2 = qconv_block(sep1, channels, kernel_size=1, strides=1, padding="valid",activation=False)
    #print(local2.shape)
    join1 = Add()([skip1,local2])
    join1 = PReLU(shared_axes=[1,2])(join1)
    #local3 = QuantLocallyConnected2D(channels,(1,1),strides=(1,1),padding="valid", implementation=1, **kwargs)(join1)
    #local3 = PReLU(shared_axes=[1,2])(local3)
    local3 = qconv_block(join1,channels,kernel_size=1,strides=1,padding="valid")
    sep2 = QuantSeparableConv2D(channels,(1,1),strides=(1,1),padding="same", **sep_kwargs)(local3)
    join2 = Add()([sep1,sep2])
    join2 = PReLU(shared_axes=[1,2])(join2)
    #join2 = None
    return join2

def embedding_setting(inputs, filters=128):
    #Z = my_qreduction(inputs, channels)
    #Z = conv_block(inputs, filters, kernel_size=5, strides=2, padding="valid") #14 x 14
    Z = conv_block(inputs, filters, kernel_size=3, strides=3, padding="valid") #4x4 
    Z = conv_block(Z, filters, kernel_size=4, strides=1, padding="valid")
    return Z

def linear_GD_conv_block(inputs, kernel_size, strides):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    Z = DepthwiseConv2D(kernel_size, strides=strides, padding="valid", depth_multiplier=1, use_bias=False)(inputs)
    Z = BatchNormalization(axis=channel_axis)(Z)
    return Z

def binaryfacenet(emb_shape=128, input_shape=(112, 112, 3), dropout=1, name="binaryfacenet", weight_file=None, include_top=False, **kwargs):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    if K.image_data_format() == "channels_first":
        X = Input(shape=(input_shape[-1], input_shape[0], input_shape[1]))
    else:
        X = Input(shape=input_shape)

    #input_shape = (112, 112, 64)
    # X = tf.random.normal(input_shape)
    #X = Input(shape=input_shape)
    Y = head_setting(X)
    Y = qconv_block(Y,64,kernel_size=3,strides=2,padding="same",groups=4) #resize to 56x56
    Y = global_feature_extraction(Y, channels=64)
    #Y = global_feature_extraction(Y, channels=64)
    #Y = conv_block(Y,64,kernel_size=3,strides=2,padding="same") #resize to 28x28
    Y = local_feature_extraction(Y, channels=64)
    Y = local_feature_extraction(Y, channels=64)
    Y = embedding_setting(Y, filters=emb_shape)
    #print(Y.shape)

    model = Model(inputs=X, outputs=Y, name=name)
    #lq.models.summary(model)
    # print("done")
    #if weight_file:
    #    model.load_weights(weight_file)
    return model



