train:
	sudo rm -r runs
	make run

run:
	python3 train.py
test:
	python3 test.py

exOx:
	python3 export_onnx.py

exRT:
	python3 export_tensorRT.py

d:
	python3 detect.py