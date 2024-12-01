FROM python:3.10.12

WORKDIR /app

# Install necessary system libraries for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0

RUN pip install --upgrade pip
#COPY requirements.txt /app/

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

#ARG AWS_ACCESS_KEY_ID
#ARG AWS_SECRET_ACCESS_KEY
#ARG AWS_DEFAULT_REGION
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_REGION=${AWS_REGION}

RUN aws s3 cp s3://cancerapp-203918887737/model.h5 /app/artifacts/models/model.h5


EXPOSE 8501

#CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#CMD ["streamlit", "run", "app.py"]
#CMD ["streamlit", "run", "--server.address", "0.0.0.0", "app.py"]
#CMD ["streamlit", "run", "--server.address", "0.0.0.0", "app.py"]
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8051", "--server.address=0.0.0.0","--server.enableCORS=false","--server.enableWebsocketCompression=false"]
#ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8051", "--server.address=0.0.0.0"]