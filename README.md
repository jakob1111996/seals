# Reimplementation of SEALS

This is a reimplementation of the paper [Similarity Search for Efficient Active Learning and Search of Rare Concepts](https://arxiv.org/pdf/2007.00077.pdf)
The goal is to recreate Figure 1b in the paper using MaxEnt-SEALS.

### Setup
I recommend setting up a new conda environment.
Install the package requirements:
```python
conda install -c pytorch faiss-gpu pip
pip install -r requirements.txt
```

### Execution
```python
python main.py
```

### Extras
Install the pre-commit hooks for linting:
```python
pre-commit install
```
