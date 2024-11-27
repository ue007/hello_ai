import onnx

# 加载ONNX模型
onnx_model = onnx.load("alexnet.onnx")

# 检查模型是否有效
onnx.checker.check_model(onnx_model)

print("ONNX模型验证通过！")
