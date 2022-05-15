import argparse

from nmt import __version__
from nmt.cli._cmd import nmt_cmd
from nmt.cli._service import nmt_service
from nmt.cli._preprocess import nmt_preprocess
from nmt.cli._download import nmt_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-v", action="version", version="%(prog)s {}".format(__version__))
    parser.set_defaults(func=lambda x: parser.print_usage())
    subparsers = parser.add_subparsers()

    # nmt cmd
    subparser_cmd = subparsers.add_parser("cmd", help="package for cmd mode")
    subparser_cmd.add_argument("package_path", help="package path")
    subparser_cmd.add_argument("--src_lang", "-s", required=True, help="src lang")
    subparser_cmd.add_argument("--tgt_lang", "-t", required=True, help="tgt lang")
    subparser_cmd.add_argument("--device", required=False, default="cpu")
    subparser_cmd.set_defaults(func=nmt_cmd)

    # nmt service
    subparser_service = subparsers.add_parser("service", help="package for service mode")
    subparser_service.add_argument("package_path", help="package path")
    subparser_service.add_argument("--src_lang", "-s", required=True, help="src lang")
    subparser_service.add_argument("--tgt_lang", "-t", required=True, help="tgt lang")
    subparser_service.add_argument("--port", "-p", required=True, help="port number")
    subparser_service.add_argument("--device", required=False, default="cpu")
    subparser_service.set_defaults(func=nmt_service)

    # nmt preprocess
    subparser_preprocess = subparsers.add_parser("preprocess", help="package for preprocess")
    subparser_preprocess.add_argument("--config_path", "-c", required=True, help="config path")
    subparser_preprocess.set_defaults(func=nmt_preprocess)

    # nmt service
    subparser_download = subparsers.add_parser("download", help="package for download")
    subparser_download.add_argument("--src_lang", "-s", required=True, help="src lang")
    subparser_download.add_argument("--tgt_lang", "-t", required=True, help="tgt lang")
    subparser_download.set_defaults(func=nmt_download)

    args = parser.parse_args()

    func = args.func

    func(args)