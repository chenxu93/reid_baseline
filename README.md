# reid_baseline
A pytorch implement

# requirements
	pretrained model:

# Train command
```
python tools/train_net.py --cfg configs/resnet50_cfg.yaml OUTPUT.SAVE_MODEL_PATH '.' SOLVER.GPU_ID 7
``` 

# Test command
python tools/test_net.py --cfg configs/resnet50_cfg.yaml TEST.WEIGHTS "trained model path" SOLVER.GPU_ID 7