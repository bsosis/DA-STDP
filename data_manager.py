
import os
import sys
import inspect
import warnings
import traceback
import functools
import pickle


def savedata(_func=None):
    """
    Save any data produced by the wrapped function, as well as all arguments.
    """
    def decorator_savedata(func):
        @functools.wraps(func)
        def wrapper_savedata(*args, save_folder=None, restore='ask', **kwargs):
            if save_folder is None:
                # Don't do anything, just run the function and return
                return func(*args, **kwargs)
            else:
                # If the data already exists, ask to load it
                if restore and (os.path.exists(os.path.join(save_folder, 'results'))
                                                or os.path.exists(os.path.join(save_folder, 'results.npz'))):
                    if restore is True or (restore == 'ask' and 
                                            input('Stored data already exists. Load old data? [Y/N]: '
                                                ).upper().strip() == 'Y'):
                        fname = os.path.join(save_folder, 'results')
                        if not os.path.exists(fname):
                            fname += '.npz'
                        try:
                            with open(fname, 'rb') as f:
                                return pickle.load(f)
                        except:
                            tb = ''.join(traceback.format_exception(*sys.exc_info()))
                            warnings.warn(f'Exception while restoring old data for {func.__name__}; will collect new data:\n{tb}')

                # Run the function
                res = func(*args, **kwargs)
            
                # Save the results as a pickle
                os.makedirs(save_folder, exist_ok=True)
                try:
                    with open(os.path.join(save_folder, 'results'), 'wb') as f:
                        pickle.dump(res, f)
                except:
                    tb = ''.join(traceback.format_exception(*sys.exc_info()))
                    warnings.warn(f'Exception while saving results of {func.__name__} using pickle; results not saved:\n{tb}')

                # Save metadata
                try:
                    # Store the function call
                    args_str = [str(a) for a in args]
                    kwargs_str = [f'{k}={v}' for k, v in kwargs.items()]
                    signature = ', '.join(args_str + kwargs_str)
                    call = f'{inspect.getmodule(func).__name__}.{func.__name__}({signature})'
                    with open(os.path.join(save_folder, 'call.txt'), 'w') as call_file:
                        call_file.write(call)
                except:
                    tb = ''.join(traceback.format_exception(*sys.exc_info()))
                    warnings.warn(f'Exception while saving metadata of {func.__name__}:\n{tb}')

                return res

        return wrapper_savedata

    if _func is None:
        return decorator_savedata
    else:
        return decorator_savedata(_func)

