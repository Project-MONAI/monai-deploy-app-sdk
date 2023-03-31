import os

from app import App

if __name__ == "__main__":
    app = App()

    # Config if need to, none for now
    app.config(os.path.join(os.path.dirname(__file__), "simple_imaging_app.yaml"))
    app.run()
