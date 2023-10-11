# -*- coding: utf-8 -*-
import click


@click.command("create-figures")
@click.argument("output_filepath", type=str)
def create_figures(output_filepath):
    """Runs data processing scripts to turn into figures."""
