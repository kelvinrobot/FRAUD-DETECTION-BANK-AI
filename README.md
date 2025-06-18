git remote add origin https://github.com/kelvinrobot/FRAUD-DETECTION-BANK-AI.git

git rm -r --cached .venv
git commit -m "Remove .venv from history"
git push origin main


uvicorn api.fraud_api:app --reload --port 8000
streamlit run streamlit_frontend/app.py


python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
