from pathlib import Path

from app import App

if __name__ == "__main__":
    app = App()

    # If there exists the named config file, use it to configure the objects which support it.
    config_file = Path(__file__).parent.absolute() / "simple_imaing_app.yaml"
    if config_file.exists():
        app.config(config_file)

    app.run()
