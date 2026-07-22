"""
Top-level argparse dispatcher for dl_etomo.

Usage:
    python -m dl_etomo <subcommand> [flags]
    dl-etomo <subcommand> [flags]          # after `pip install -e .`

Run `dl-etomo --help` or `dl-etomo <subcommand> --help` for details.
"""

from .cli import build_parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
