
setup: 
	source venv/bin/activate
	pip install -r requirements.txt
t:
	python3 train.py

p:
	python3 predict.py

bonus:t
	python3 display-data.py

clean:
	rm -f *.json *.png

.PHONY: train predict precision clean
