
FROM python:3.10-slim


WORKDIR /interface


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


EXPOSE 8501


CMD ["streamlit", "run", "interface.py", "--server.port=8501", "--server.address=0.0.0.0"]