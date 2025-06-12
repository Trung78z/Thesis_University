d:
	python3 prediction/detect.py

training:
	python train_test/train.py

exRT:
	python train_test/export_tensorRT.py

test:
	python test.py

read:
	python tests/read_video.py

check_edge:
	python prediction/check_edge.py

init:
	python3 -m venv .env --system-site-packages
init-pip:
	.env/bin/python3 -m pip install --upgrade pip
	.env/bin/python3 -m pip install -r requirements.txt