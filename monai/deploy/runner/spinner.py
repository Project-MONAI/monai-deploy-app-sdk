import itertools
import sys
from multiprocessing import Event, Lock, Process


class ProgressSpinner:
    """
    Progress spinner for console.
    """

    def __init__(self, message, delay=0.2):
        self.spinner_symbols = itertools.cycle(['-', '/', '|', '\\'])
        self.delay = delay
        self.stop_event = Event()
        self.spinner_visible = False
        sys.stdout.write(message)

    def __enter__(self):
        self.start()

    def __exit__(self, exception, value, traceback):
        self.stop()

    def _spinner_task(self):
        while not self.stop_event.wait(self.delay):
            self._remove_spinner()
            self._write_next_symbol()

    def _write_next_symbol(self):
        with self._spinner_lock:
            if not self.spinner_visible:
                sys.stdout.write(next(self.spinner_symbols))
                self.spinner_visible = True
                sys.stdout.flush()

    def _remove_spinner(self, cleanup=False):
        with self._spinner_lock:
            if self.spinner_visible:
                sys.stdout.write('\b')
                self.spinner_visible = False
                if cleanup:
                    # overwrite spinner symbol with whitespace
                    sys.stdout.write(' ')
                    sys.stdout.write('\r')
                sys.stdout.flush()

    def start(self):
        """
        Start spinner as a separate process.
        """
        if sys.stdout.isatty():
            self._spinner_lock = Lock()
            self.stop_event.clear()
            self.spinner_process = Process(target=self._spinner_task)
            self.spinner_process.start()

    def stop(self):
        """
        Stop spinner process.
        """
        if sys.stdout.isatty():
            self.stop_event.set()
            self._remove_spinner(cleanup=True)
            self.spinner_process.join()
            sys.stdout.write('\n')
        else:
            sys.stdout.write('\r')
