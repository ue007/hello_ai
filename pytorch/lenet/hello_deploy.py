import onnxruntime
import torch
import numpy as np
import matplotlib.pyplot as plt

# 定义一个函数来加载ONNX模型
def load_onnx_model(model_path):
    session = onnxruntime.InferenceSession(model_path)
    return session

# 定义一个函数来对数据进行预处理（以MNIST数据为例）
def preprocess_data(data):
    # 将数据转换为float32类型，并进行归一化
    data = data.astype(np.float32)
    data = (data - 0.1307) / 0.3081
    # 添加批次维度（如果数据本身没有批次维度）
    if len(data.shape) == 3:
        data = np.expand_dims(data, axis = 0)
    return data

# 生成一个随机的MNIST格式的图像数据（示例）
random_image = torch.randn(1, 1, 28, 28)
preprocessed_image = preprocess_data(random_image.numpy())

# 加载ONNX模型
onnx_session = load_onnx_model("./save_model/lenet_model.onnx")

# 获取模型的输入和输出名称
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

# 进行推理
result = onnx_session.run([output_name], {input_name: preprocessed_image})
predicted_label = np.argmax(result[0])

print("预测的数字标签:", predicted_label)