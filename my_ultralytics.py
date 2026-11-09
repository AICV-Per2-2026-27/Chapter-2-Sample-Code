import ultralytics
import torch

ultralytics.checks()
mps = torch.device('mps')

model = ultralytics.YOLO('yolo26m.pt')

results = model.train(
        data='my_data.yaml', 
        device=mps, 
        epochs=300, 
        optimizer='Adam', 
        lr0=0.005, 
        lrf=0.0001, 
        momentum=0.9, 
        save_period=10,
        workers=16,
        patience=0,
)