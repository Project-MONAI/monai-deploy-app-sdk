from app import AISpleenSegApp

from monai.deploy.logger import load_env_log_level

if __name__ == "__main__":
    load_env_log_level()
    AISpleenSegApp().run()
