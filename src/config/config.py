from dotenv import load_dotenv
import os
import json

load_dotenv()
host = os.getenv("HOST")
port = os.getenv("PORT")
database = os.getenv("DATABASE")
user = os.getenv("USER")
password = os.getenv("PASSWORD")

db_params = {
    "host": host,
    "port": port,
    "database": database,
    "user": user,
    "password": password,
}

backtesting_config = {}
with open("parameter/backtesting_parameter.json", "r") as bf:
    backtesting_config = json.load(bf)

optimization_params = {}
with open("parameter/optimization_parameter.json", "r") as of:
    optimization_params = json.load(of)