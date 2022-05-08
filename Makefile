.PHONY: init clear rebuild purge test lint lab ipython bench

init: purge
	python -m venv env
	env/bin/pip install --upgrade pip wheel setuptools numpy
	env/bin/pip install .[test]

clear:
	-env/bin/pip uninstall -y pyflatbush
	-@rm -rf .pytest_cache tests/__pycache__ __pycache__ pyflatbush/__pycache__ dist .coverage
	-@find . -type d -name '*.egg-info' | xargs rm -r
	-@find . -type f -name '*.pyc' | xargs rm -r
	-@find . -type d -name '*.ipynb_checkpoints' | xargs rm -r

rebuild: clear
	env/bin/pip install .[test]

purge: clear
	-@rm -rf env

bench:
	cd bench && ../env/bin/python bench.py

test:
	env/bin/pytest tests/*

ipykernel:
	env/bin/pip install ipykernel
	env/bin/python -m ipykernel install --user --name "pyflatbush"
