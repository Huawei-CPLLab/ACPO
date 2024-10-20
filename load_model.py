import torch
# import torch.nn
# import torch.functional
from ACPOmodel.src.models import NetFC

model = NetFC()
# model = torch.load("/Users/lvyinrun/workspace/openEuler/ACPO_zm/ACPO-model/src/log/inline_modelV0.0/modelfi.pth", map_location=torch.device('cpu'))
# model = torch.load("/Users/lvyinrun/workspace/openEuler/ACPO_zm/ACPOmodel/src/log/inline_modelV0.0/modelfi.pth")
state_dict = torch.load("/Users/lvyinrun/workspace/openEuler/ACPO_zm/ACPOmodel/src/log/inline_modelV0.0/modelfi.pth", weights_only=False)
model.load_state_dict(state_dict)
print(model)
