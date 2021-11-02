import torch.onnx
import torch
import torchvision
from glob import glob
# Standard ImageNet input - 3 channels, 224x224,
# values don't matter as we care about network structure.
# But they can also be real inputs.
dummy_input = torch.randn(1, 1, 320, 320).cpu()
# Obtain your model, it can be also constructed in your script explicitly
# # model = torchvision.models.alexnet(pretrained=True)
# if glob("D:\code\python_code\classify\efficientnet-pytorch\dataset\model\*.1pth") :
#     print(1000)
# else:
#     print(1)




model = torch.load("model/weight3.pth")


# Invoke export
torch.onnx.export(model, dummy_input, "weight2.onnx",example_outputs=(1,9),training=False)


# import onnx
#
# # Load the ONNX model
# model = onnx.load("alexnet.onnx")
#
# # Check that the IR is well formed
# onnx.checker.check_model(model)
#
# # Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))

#
# import numpy as np  # we're going to use numpy to process input and output data
# import onnxruntime  # to inference ONNX models, we use the ONNX Runtime
# import time
# import json
# from PIL import Image
#
#
# def load_labels(path):
#     with open(path) as f:
#         data = json.load(f)
#     return np.asarray(data)
#
#
# # 图像预处理
# def preprocess(input_data):
#     # convert the input data into the float32 input
#     img_data = input_data.astype('float32')
#
#     # normalize
#     mean_vec = np.array([0.485, 0.456, 0.406])
#     stddev_vec = np.array([0.229, 0.224, 0.225])
#     norm_img_data = np.zeros(img_data.shape).astype('float32')
#     for i in range(img_data.shape[0]):
#         norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
#
#     # add batch channel
#     norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
#     return norm_img_data
#
#
# def softmax(x):
#     x = x.reshape(-1)
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)
#
#
# def postprocess(result):
#     return softmax(np.array(result)).tolist()
#
#
# # Load the raw image
# img = Image.open(r"D:\code\test_image\1.jpg")
# img = img.resize((224, 224), Image.BILINEAR)
# print("Image size: ", img.size)
#
# image_data = np.array(img).transpose(2, 0, 1)
# input_data = preprocess(image_data)
#
# # Run the model on the backend
# session = onnxruntime.InferenceSession('.//alexnet.onnx', None)
#
# # get the name of the first input of the model
# input_name = session.get_inputs()[0].name
# print('Input Name:', input_name)
#
# # Inference
# start = time.time()
# raw_result = session.run([], {input_name: input_data})
# end = time.time()
# res = postprocess(raw_result)
#
# inference_time = np.round((end - start) * 1000, 2)
# idx = np.argmax(res)
#
# print('========================================')
# print('Final top prediction is: %d' % idx)
# print('========================================')
#
# print('========================================')
# print('Inference time: ' + str(inference_time) + " ms")
# print('========================================')


# import os
# import tensorflow as tf
# model_dir = r"D:\code\python_code\test\convert"
# model_name = "frozen_model.pb"

# def create_graph():
#     with tf.gfile.FastGFile(os.path.join(model_dir, model_name), 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name='')
#
#
# for node in tf.get_default_graph().as_graph_def().node :
# 	if node.op == 'RefSwitch':
# 		node.op = 'Switch'
# 		for index in range(len(node.input)):
# 			if 'moving_' in node.input[index]:
# 				node.input[index] = node.input[index] + '/read'
# 	elif node.op == 'AssignSub':
# 		node.op = 'Sub'
# 		if 'use_locking' in node.attr:
# 			del node.attr['use_locking']
#
# create_graph()
# tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
# for tensor_name in tensor_name_list:
#   	print(tensor_name, '\n')
# import tensorflow as tf
#
# from tensorflow.python.platform import gfile
#
# model_path = "model/model_1.pb"
#
# # read pb into graph_def
# # with tf.gfile.GFile(model_path, "rb") as f:
# # 	graph_def = tf.GraphDef()
# # 	graph_def.ParseFromString(f.read())
# #
# # # import graph_def
# # with tf.Graph().as_default() as graph:
# # 	tf.import_graph_def(graph_def)
#
#
# # read graph definition
# f = gfile.FastGFile(model_path, "rb")
# gd = graph_def = tf.GraphDef()
# graph_def.ParseFromString(f.read())
#
# # fix nodes
# for node in graph_def.node:
#     if node.op == 'RefSwitch':
#
#         node.op = 'Switch'
#         # for index in range(len(node.input)):
#         #     if 'moving_' in node.input[index]:
#         #         node.input[index] = node.input[index] + '/read'
#     elif node.op == 'AssignSub':
#         node.op = 'Sub'
#         if 'use_locking' in node.attr: del node.attr['use_locking']
#     elif node.op == 'AssignAdd':
#         node.op = 'Add'
#         if 'use_locking' in node.attr: del node.attr['use_locking']
#     elif node.op == 'Assign':
#         node.op = 'Identity'
#         if 'use_locking' in node.attr: del node.attr['use_locking']
#         if 'validate_shape' in node.attr: del node.attr['validate_shape']
#         # if len(node.input) == 2:
#         # # input0: ref: Should be from a Variable node. May be uninitialized.
#         # # input1: value: The value to be assigned to the variable.
#         #     node.input[0] = node.input[1]
#         #     del node.input[1]
#
#
# # import graph into session
# tf.import_graph_def(graph_def, name='')
# tf.train.write_graph(graph_def, './', 'good_frozen.pb', as_text=False)
# # tf.train.write_graph(graph_def, './', 'good_frozen.pbtxt', as_text=True)