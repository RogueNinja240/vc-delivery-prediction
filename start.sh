#!/bin/bash

# Start FastAPI in the background (the '&' makes it run in the background)
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in the foreground. 
# We use Render's default $PORT, or 8501 if running locally.
PORT=${PORT:-8501}
streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0