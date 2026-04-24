"""Command-line interface for slocum-tpw."""

import argparse
import importlib
import logging
import sys

from slocum_tpw import __version__

# Subcommand registry: name -> (module path, help text).
# Modules are imported lazily so cold-start latency for lightweight subcommands
# (decode-argos, log-harvest) does not pay for matplotlib / scipy / gsw.
_SUBCOMMANDS = {
    "decode-argos": (
        "slocum_tpw.decode_argos",
        "Decode ARGOS satellite messages into NetCDF",
    ),
    "log-harvest": (
        "slocum_tpw.log_harvest",
        "Harvest Slocum glider log files into NetCDF",
    ),
    "mk-combined": (
        "slocum_tpw.mk_combined",
        "Combine glider data into CF-compliant NetCDF",
    ),
    "recover-by": (
        "slocum_tpw.recover_by",
        "Estimate glider recovery date from battery decay",
    ),
    "simulate-leak": (
        "slocum_tpw.simulate_leak",
        "Simulate sealed-body vacuum observations for leak-detection testing",
    ),
    "analyze-leak": (
        "slocum_tpw.analyze_leak",
        "Estimate d(n/V)/dt and its uncertainty from sealed-body observations",
    ),
}


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
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="slocum-tpw",
        description="TWR Slocum glider data processing utilities",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_logging_args(parser)

    subparsers = parser.add_subparsers(dest="command")

    # Pre-scan argv for the requested subcommand so we only import its module.
    # Top-level flags (--version, --verbose, --debug) take no values, so the
    # first argv entry that matches a known subcommand name is the subcommand.
    invoked = next((a for a in argv if a in _SUBCOMMANDS), None)

    if invoked is not None:
        module_name, help_text = _SUBCOMMANDS[invoked]
        module = importlib.import_module(module_name)
        sub = subparsers.add_parser(invoked, help=help_text)
        module.add_arguments(sub)
        sub.set_defaults(func=module.run)
    else:
        # No subcommand on the command line: register lightweight stubs so
        # --help still lists every subcommand.
        for name, (_, help_text) in _SUBCOMMANDS.items():
            subparsers.add_parser(name, help=help_text)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(2)

    _configure_logging(args)
    sys.exit(args.func(args))
