
setup:
	python3 -m venv venv

venv:
	source venv/bin/activate

install:
	venv/bin/python3 -m pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
t:
	python3 train.py

p:
	python3 predict.py

bonus:t
	python3 display-data.py

clean:
	rm -rf venv
	rm -f *.json *.png

.PHONY: train predict precision clean
