# pyrefly: ignore [missing-import]
import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Memory:", torch.cuda.get_device_properties(0).total_memory)