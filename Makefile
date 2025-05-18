d:
	python prediction/detect.py

training:
	sudo rm -rf runs
	python train/train.py

exOx:
	python train/export_onnx.py

exRT:
	python train/export_tensorRT.py

test:
	python test.py
