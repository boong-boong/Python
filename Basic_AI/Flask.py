#가상환경 설치 할것
from flask import Flask
import sys
application = Flask(__name__)


@application.route("/")
def hello():
    return "Hello Flask!"

@application.route("/1")
def One():
    return "{msg:'1 page'}"

@application.route("/2")
def Two():
    return "{msg:'2 page'}"



if __name__ == "__main__":
    application.run(debug=True)