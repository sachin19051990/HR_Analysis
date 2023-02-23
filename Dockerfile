FROM python
RUN pip install --upgrade pip
WORKDIR /app
COPY . /app
EXPOSE 8501
RUN pip install -r requirements.txt
CMD streamlit run server.py