import argparse
from src.util import package_application

def package_application_wrapper(args):
    package_application(args)

def main():
    parser = argparse.ArgumentParser(description='MONAI Application Package Builder')
    action_subparser = parser.add_subparsers(help='action')

    build_parser = action_subparser.add_parser(
        'build', help='Build MONAI Application Package')
    generate_parser = action_subparser.add_parser(
        'generate', help='Generate Configuration Files for MONAI Application Package')

    build_parser.add_argument('application', type=str, help="MONAI application path")
    build_parser.add_argument('--name', type=str, help="MONAI application package name")
    build_parser.add_argument('--base', type=str, help="Base Application Image")
    build_parser.add_argument('--app-config', type=str, help="Application Manifest Path")
    build_parser.add_argument('--working-dir','-w', type=str, help="Directory mounted in container for Application")
    build_parser.add_argument('--input','-i', type=str, help="Directory mounted in container for Application Input")
    build_parser.add_argument('--output','-o', type=str, help="Directory mounted in container for Application Output")
    
    # SDK provided values
    build_parser.add_argument('--models','-m', type=str, help="Models Path")
    # build_parser.add_argument('--version','-m', type=str, help="App Version")
    # build_parser.add_argument('--app-version','-m', type=str, help="App Version")
    # build_parser.add_argument('--command','-m', type=str, help="Driver Command")
    # build_parser.add_argument('--resource','-m', type=str, help="Resource")

    build_parser.set_defaults(func=package_application_wrapper)

    args = parser.parse_args()
    try:
        func = args.func
    except AttributeError:
        parser.error("too few arguments")
    func(args)


if __name__ == "__main__":
    main()
