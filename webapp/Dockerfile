FROM python:3.8

COPY . /webapp

WORKDIR /webapp

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]