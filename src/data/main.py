# -*- coding: utf-8 -*-
import logging

import click
from bloomberg.ai.nlu.remoteio import RemoteIO

from bloomberg.textqa.evaluating_itr_in_mm.data.generate import generate_data


@click.command("create-data")
@click.argument("input_filepath", type=str)
@click.argument("output_filepath", type=str)
def create_dataset(input_filepath, output_filepath):
    """Generates processed data from raw data."""
    logger = logging.getLogger(__name__)
    # RemoteIO let's you seamlessly switch between localFS/HDFS/S3
    # Docs available at https://bbgithub.dev.bloomberg.com/nlu/remoteio
    # See project readme for the DSP Tips talk featuring remoteio
    # Example snippet on how remote io works
    if RemoteIO.exists(input_filepath):
        data = generate_data({})
        # Do not use print! Use logging instead!
        logger.debug("Generated data: %r", data)
