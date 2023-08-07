FROM python:3.7.13
#FROM tensorflow/serving

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

CMD [ "python", "app.py" ]

EXPOSE 5003