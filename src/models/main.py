# -*- coding: utf-8 -*-
import os
import random

import click
from bloomberg.ds.maestro import client

from bloomberg.textqa.evaluating_itr_in_mm.models.train import train_model


@click.command("create-model")
@click.argument("input_filepath", type=str)
@click.argument("output_filepath", type=str)
@click.option("--hypertune", type=str, default="Test")
def create_model(input_filepath, output_filepath, hypertune):
    """"Trains and creates a model."""	    """"Trains and creates a model."""
    print(os.environ.get("PAR_GREETING", "Test Another"))
    print(hypertune)
    train_model({})
    run_id = os.environ.get('HYPERTUNE_RUN_ID')
    maestro_endpoint = os.environ.get('HYPERTUNE_ENDPOINT')
    # Replace the value here with a metric denoting the performance of your model.
    value = random.random()
    # Register the results with maestro so that it picks the best model
    if maestro_endpoint and run_id:
        client.register_run_result(run_id=run_id, url=maestro_endpoint, value=value)
