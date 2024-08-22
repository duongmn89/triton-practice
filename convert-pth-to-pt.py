#This script aims to convert pytorch .pth model to .pt so we can load it to triton
import torch
from utils.model import STRModel

# Create PyTorch Model Object
model = STRModel(input_channels=1, output_channels=512, num_classes=37)

# Load model weights from external file
state = torch.load("<model-name>.pth")
state = {key.replace("module.", ""): value for key, value in state.items()}
model.load_state_dict(state)

# Create pt file by tracing model
trace_input = torch.randn(1, 1, 32, 100)
traced_script_module = torch.jit.trace(model, trace_input)
traced_script_module.save("model.pt")
