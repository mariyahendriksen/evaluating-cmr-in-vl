"""Entry point for evaluating_itr_in_mm CLI."""
import logging

import click

from bloomberg.textqa.evaluating_itr_in_mm.data.main import create_dataset
from bloomberg.textqa.evaluating_itr_in_mm.features.main import create_features
from bloomberg.textqa.evaluating_itr_in_mm.models.main import create_model
from bloomberg.textqa.evaluating_itr_in_mm.visualization.main import create_figures


@click.group()
def cli() -> None:
    """Entry point for evaluating_itr_in_mm."""


cli.add_command(create_dataset)
cli.add_command(create_features)
cli.add_command(create_model)
cli.add_command(create_figures)


if __name__ == "__main__":
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    cli()
