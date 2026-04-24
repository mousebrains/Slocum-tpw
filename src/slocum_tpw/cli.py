"""Command-line interface for slocum-tpw."""

import argparse
import logging
import sys

from slocum_tpw import __version__


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add --verbose and --debug flags to an argument parser."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verbose", action="store_true", help="Enable verbose output")
    group.add_argument("--debug", action="store_true", help="Enable debug output")


def _configure_logging(args: argparse.Namespace) -> None:
    """Configure logging based on parsed arguments."""
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        force=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the slocum-tpw CLI."""
    parser = argparse.ArgumentParser(
        prog="slocum-tpw",
        description="TWR Slocum glider data processing utilities",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_logging_args(parser)

    subparsers = parser.add_subparsers(dest="command")

    # -- decode-argos --
    from slocum_tpw.decode_argos import add_arguments as add_da_args
    from slocum_tpw.decode_argos import run as run_da

    da_parser = subparsers.add_parser(
        "decode-argos", help="Decode ARGOS satellite messages into NetCDF"
    )
    add_da_args(da_parser)
    da_parser.set_defaults(func=run_da)

    # -- log-harvest --
    from slocum_tpw.log_harvest import add_arguments as add_lh_args
    from slocum_tpw.log_harvest import run as run_lh

    lh_parser = subparsers.add_parser(
        "log-harvest", help="Harvest Slocum glider log files into NetCDF"
    )
    add_lh_args(lh_parser)
    lh_parser.set_defaults(func=run_lh)

    # -- mk-combined --
    from slocum_tpw.mk_combined import add_arguments as add_mc_args
    from slocum_tpw.mk_combined import run as run_mc

    mc_parser = subparsers.add_parser(
        "mk-combined", help="Combine glider data into CF-compliant NetCDF"
    )
    add_mc_args(mc_parser)
    mc_parser.set_defaults(func=run_mc)

    # -- recover-by --
    from slocum_tpw.recover_by import add_arguments as add_rb_args
    from slocum_tpw.recover_by import run as run_rb

    rb_parser = subparsers.add_parser(
        "recover-by", help="Estimate glider recovery date from battery decay"
    )
    add_rb_args(rb_parser)
    rb_parser.set_defaults(func=run_rb)

    # -- simulate-leak --
    from slocum_tpw.simulate_leak import add_arguments as add_sl_args
    from slocum_tpw.simulate_leak import run as run_sl

    sl_parser = subparsers.add_parser(
        "simulate-leak",
        help="Simulate sealed-body vacuum observations for leak-detection testing",
    )
    add_sl_args(sl_parser)
    sl_parser.set_defaults(func=run_sl)

    # -- analyze-leak --
    from slocum_tpw.analyze_leak import add_arguments as add_al_args
    from slocum_tpw.analyze_leak import run as run_al

    al_parser = subparsers.add_parser(
        "analyze-leak",
        help="Estimate d(n/V)/dt and its uncertainty from sealed-body observations",
    )
    add_al_args(al_parser)
    al_parser.set_defaults(func=run_al)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(2)

    _configure_logging(args)
    sys.exit(args.func(args))
