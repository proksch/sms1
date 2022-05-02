FROM python:3.7.13-slim

WORKDIR /root/
COPY requirements.txt .

RUN mkdir output &&\
	python -m pip install --upgrade pip &&\
	pip install -r requirements.txt

COPY src src
COPY smsspamcollection smsspamcollection

RUN python src/read_data.py &&\
	python src/text_preprocessing.py &&\
	python src/text_classification.py

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["src/serve_model.py"]

