#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt

python3 train_model.py
python3 main.py
