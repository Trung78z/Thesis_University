train:
	sudo rm -r runs
	make run

run:
	python train.py
test:
	python test.py

exOx:
	python export_onnx.py

exRT:
	python export_tensorRT.py