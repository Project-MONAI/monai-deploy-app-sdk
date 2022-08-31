import logging
import shutil
import traceback
from pathlib import Path

from app import AIUnetrSegApp

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # This main function is an example to show how a batch of input can be processed.
    # It assumes that in the app input folder there are a number of subfolders, each
    # containing a discrete input to be processed. Each discrete payload can have
    # multiple DICOM instances file, optionally organized in its own folder structure.
    # The application object is first created, and on its init the model network is
    # loaded as well as pre and post processing transforms. This app object is then
    # run multiple times, each time with a single discrete payload.

    app = AIUnetrSegApp(do_run=False)

    # Preserve the application top level input and output folder path, as the path
    # in the context may change on each run if the I/O arguments are passed in.
    app_input_path = Path(app.context.input_path)
    app_output_path = Path(app.context.output_path)

    # Get subfolders in the input path, assume each one contains a discrete payload
    input_dirs = [path for path in app_input_path.iterdir() if path.is_dir()]

    # Set the output path for each run under the app's output path, and do run
    work_dirs = []
    for idx, dir in enumerate(input_dirs):
        try:
            output_path = app_output_path / f"{dir.name}_output"
            work_dir = f".unetr_app_workdir{idx}"
            work_dirs.extend(work_dir)

            logging.info(f"Start processing input in: {dir} with results in: {output_path}")

            # Run app with specific input and output path.
            # Passing in the input and output do have the side effect of changing
            # app context. This side effect will likely be eliminated in later releases.
            app.run(input=dir, output=output_path, workdir=work_dir)

            logging.info(f"Completed processing input in: {dir} with results in: {output_path}")
        except Exception as ex:
            logging.error(f"Failed processing input in {dir}, due to: {ex}\n")
            traceback.print_exc()
        finally:
            # Remove the workdir; alternatively do this later, if storage space is not a concern.
            shutil.rmtree(work_dir)

    # Alternative. Explicitly remove the working dirs at the end of main.
    # [shutil.rmtree(work_dir, ignore_errors=True) for work_dir in work_dirs]
