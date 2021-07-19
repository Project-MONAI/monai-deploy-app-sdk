import argparse
from src.util import package_application

def package_application_wrapper(args):
    package_application(args.name, args.base,args.entrypoint,args.app_config,args.package_config)

def main():
    parser = argparse.ArgumentParser(description='MONAI Application Package Builder')
    action_subparser = parser.add_subparsers(help='action')

    build_parser = action_subparser.add_parser(
        'build', help='Build MONAI Application Package')
    generate_parser = action_subparser.add_parser(
        'generate', help='Generate Configuration Files for MONAI Application Package')

    build_parser.add_argument('name', type=str, help="MONAI application package name")
    build_parser.add_argument('--base', type=str, help="Base Application Image")
    build_parser.add_argument('--entrypoint', type=str, help="Base Application Image Entrypoint")
    build_parser.add_argument('--app-config', type=str, help="Application Manifest Path")
    build_parser.add_argument('--package-config', type=str, help="Package Manifest Path")
    build_parser.set_defaults(func=package_application_wrapper)

    args = parser.parse_args()
    try:
        func = args.func
    except AttributeError:
        parser.error("too few arguments")
    func(args)


if __name__ == "__main__":
    main()
