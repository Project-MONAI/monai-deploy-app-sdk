import argparse
from src.util import package_application

def package_application_wrapper(args):
    package_application(args)

def main():
    parser = argparse.ArgumentParser(description='MONAI Application Package Builder')
    action_subparser = parser.add_subparsers(help='action')

    package_parser = action_subparser.add_parser(
        'package', help='Package MONAI Application Package')
    generate_parser = action_subparser.add_parser(
        'generate', help='Generate Configuration Files for MONAI Application Package')

    package_parser.add_argument('application', type=str, help="MONAI application path")
    package_parser.add_argument('--tag', '-t', type=str, help="MONAI application package tag")
    package_parser.add_argument('--base', type=str, help="Base Application Image")
    package_parser.add_argument('--app-config', type=str, help="Application Manifest Path")
    package_parser.add_argument('--working-dir','-w', type=str, help="Directory mounted in container for Application")
    package_parser.add_argument('--input','-i', type=str, help="Directory mounted in container for Application Input")
    package_parser.add_argument('--output','-o', type=str, help="Directory mounted in container for Application Output")
    package_parser.add_argument('--models','-m', type=str, help="Directory mounted in container for Models Path")
    package_parser.add_argument('--verbose', action='store_true', help="Display debug output")

    # SDK provided values
    package_parser.add_argument('--params','-p', type=str, help="SDK Parameters")
    # package_parser.add_argument('--version','-m', type=str, help="App Version")
    # package_parser.add_argument('--app-version','-m', type=str, help="App Version")
    # package_parser.add_argument('--command','-m', type=str, help="Driver Command")
    # package_parser.add_argument('--resource','-m', type=str, help="Resource")

    package_parser.set_defaults(func=package_application_wrapper)

    args = parser.parse_args()

    # Required Arguements
    if not args.tag:
        parser.error("required arguement '--tag','-t' image tag is missing")
        exit()

    try:
        func = args.func
    except AttributeError:
        parser.error("too few arguments")
    func(args)


if __name__ == "__main__":
    main()
