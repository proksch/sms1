# SMS Spam Detection Using Machine Learning

This project is adapted from: https://github.com/rohan8594/SMS-Spam-Detection

## Instructions for Compiling

a) Clone repo.

```
$ git clone https://github.com/rohan8594/SMS-Spam-Detection.git
$ cd SMS-Spam-Detection
```

b) Install all dependencies.

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

c) Run various scripts

```
$ python src/read_data.py
$ python src/text_preprocessing.py
$ python src/text_classification.py
$ python src/parameter_tuning.py
$ python src/learning_curve.py
$ python src/check_bias.py
$ python src/address_imbalance.py
```

d) Serve the model as a REST API

```
$ python src/serve_model.py
```

You can test the API using the following:

```
curl -X POST "http://127.0.0.1:8080/predict" -H  "accept: application/json" -d "{sms: hello world!}"
```

Alternatively, you can access the UI using your browser: http://127.0.0.1:8080/apidocs
