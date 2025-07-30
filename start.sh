#!/bin/bash

python app/main_api.py &

streamlit run app/app_streamlit.py --server.port=8501 --server.address=0.0.0.0
