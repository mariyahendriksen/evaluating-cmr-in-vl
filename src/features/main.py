# -*- coding: utf-8 -*-
import click

from bloomberg.textqa.evaluating_itr_in_mm.features.generate import generate_features


@click.command("create-features")
@click.argument("input_filepath", type=str)
@click.argument("output_filepath", type=str)
def create_features(input_filepath, output_filepath):
    """"Generate features"""
    generate_features({})
