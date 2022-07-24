"""
Local library for shared functions.
"""
from rich.console import Console


def batch_routine(func, name_to_file_path_dict):
    """
    Run a function on a collections of files.
    Collect all exceptions along the way.
    """
    all_success = True
    console = Console()

    for _name, _path in name_to_file_path_dict.items():
        console.print(f"==== Running {func.__name__} on {_name} ====")
        _script, _process = func(_name, _path)
        _success = _process.returncode == 0
        all_success = all_success and _success

        if not _success:
            console.print(
                f"!!!! Error from {func.__name__} on {_name} !!!!", style="red bold"
            )
            console.print(f"{_script}\n\n", style="blue")
            console.print(f"{_process.stderr}\n\n", style="red")

    if not all_success:
        raise RuntimeError("Script test failed.")
