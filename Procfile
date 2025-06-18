# Procfile (for Digital Ocean App Platform)
web: gunicorn app:app

# runtime.txt (specify Python version)
python-3.11.0

# .gitignore
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
.venv
.env
*.db
*.sqlite3
*.log
.DS_Store
.vscode/
*.pkl
models/saved/*.pkl

# app.yaml (optional - for Google Cloud App Engine)
runtime: python311

env_variables:
  FLASK_ENV: production

# For Docker deployment (optional)
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]