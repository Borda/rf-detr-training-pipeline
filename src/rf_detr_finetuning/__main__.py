"""CLI entry point for the RF-DETR training pipeline."""

import logging

from rf_detr_finetuning.cli import commands

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(commands, as_positional=False)
