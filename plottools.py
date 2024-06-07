
import os

import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, save_folder=None, silent=False):
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder
        self.silent = silent

        # https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
        if silent:
            plt.switch_backend('Agg')
        else:
            plt.switch_backend('TkAgg')

    def __enter__(self):
        return self

    def register(self, fig, fname, formats=None, **kwargs):
        if self.save_folder is not None:
            if formats is None:
                fig.savefig(os.path.join(self.save_folder, fname), **kwargs)
            else:
                for fmt in formats:
                    fig.savefig(os.path.join(self.save_folder, fname + '.' + fmt), **kwargs)

        if self.silent:
            plt.close(fig)


    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None: # An exception was raised
            plt.close('all')
        else:
            if not self.silent:
                plt.show()
            else:
                plt.close('all')
        # Note: returns None, which is falsy
        # This makes the 'with' statement reraise any exception as it's not handled here
        