#!/bin/bash
set -e
python3 -m pip install -r requirements.txt
python3 src/eda.py
python3 src/train_models.py
streamlit run app/app.py
