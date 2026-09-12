from src.pipelines.prediction_pipeline import PredictionPipeline
import time
import pywebio
from pywebio.input import *
from pywebio.output import *
import numpy as np

@pywebio.config(title = "Term Deposit Prediction")
def buildInterface():
    # building the interface
    data = input_group("Fill the details below", [
        select(
            label = "Select the type of job",
            options = [
                "services", "blue-collar", "technician", "management", "student", "self-employed",\
                    "housemaid", "retired", "entrepreneur", "admin.", "unemployed"
            ],
            required = True,
            name = "job"
        ),
        select(
            label = "Select month",
            options = [
                "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
            ],
            required = True,
            name = "month"
        ),
        input("Enter average yearly balance (in euros)", name = "balance", required = True, type = FLOAT),
        input("Enter last contact duration (in seconds)", name = "duration", required = True, type = NUMBER),
        input("Enter number of contacts performed for this customer in current campaign", name = "campaign",\
            required = True, type = NUMBER),
        radio("Select outcome of the previous campaign for this customer", options = ["success", "failure", "not applicable"], required = True,\
            name = "poutcome", inline = True),
        radio("Was the customer ever contacted previously", options = ["yes", "no"], inline = True,
            required = True, name = "contacted_previously"),
        select("Type of loan(s) the customer has/have", options = ["personal loan only", "housing loan only", "both", "none"],\
            required = True, name = "loans")
    ])

    # encoding few features manually
    ### poutcome
    if data["poutcome"] == "success":
        data["poutcome"] = 1
    else: data["poutcome"] = 0

    ### contacted_previously
    if data["contacted_previously"] == "yes":
        data["contacted_previously"] = 1
    else: data["contacted_previously"] = 0

    ### loans
    if data["loans"] == "personal loan only":
        data["loans"] = 1
    elif data["loans"] == "housing loan only":
        data["loans"] = 1
    elif data["loans"] == "both":
        data["loans"] = 2
    else: data["loans"] = 0

    with put_loading().style("position: absolute; left: 50%; top: 50%"):
        time.sleep(2)
        predictionPipeline = PredictionPipeline()
        result = predictionPipeline.predictResult(array = np.array(list(data.values())))

    put_image("https://www.pnbmetlife.com/content/dam/pnb-metlife/images/articles/savings/five-interesting-facts.jpg")
    if result == 0:
        put_text("\n\nMODEL PREDICTION : The person is not likely to buy a term deposit.")
    else:
        put_text("\n\nMODEL PREDICTION : The person is expected to buy the term deposit.")

if __name__ == "__main__":
    pywebio.platform.flask.start_server(buildInterface, port = 80, host = "0.0.0.0")
