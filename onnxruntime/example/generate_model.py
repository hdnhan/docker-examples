import torch
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(pretrained=True)
model.to(device)
model.eval()

batch_size = 1
h, w = 512, 512
x = torch.rand((batch_size, 3, h, w), device=device)

# ONNX model
torch.onnx.export(
    model=model,
    args=x,
    f="/model.onnx",
    export_params=True,
    verbose=False,
    opset_version=11,
    training=torch.onnx.TrainingMode.EVAL,
    input_names = ['input'],   
    output_names = ['output'],
)
