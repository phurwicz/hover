"""
Module for controlling the configuration of hover itself.
"""
import re
import configparser


def auto_interpret(text):
    """
    Automatic interpretation of a string.
    """
    if not isinstance(text, str):
        return text

    if re.search(r"^\-?\d+$", text):
        return int(text)
    if re.search(r"^\-?\d+\.\d+$", text):
        return float(text)
    if re.search(r"(?i)^(yes|on|true)$", text):
        return True
    if re.search(r"(?i)^(no|off|false)$", text):
        return False
    return text


class LockableConfig:
    """
    Dict-like object where key reads locks the value from updates.
    """

    def __init__(self, data_dict):
        self._data = data_dict
        self.open = {_k: True for _k in self._data.keys()}

    def __getitem__(self, key):
        value = self._data[key]
        self.open[key] = False
        return value

    def __setitem__(self, key, value):
        self.assert_open(key)
        if key not in self.open:
            self.open[key] = True
        self._data[key] = value

    def assert_open(self, key):
        assert self.open.get(key, True), f"{key} is locked from updates."

    def update(self, data_dict):
        """
        Check all the key lock statuses, then update.
        """
        for _k in data_dict.keys():
            self.assert_open(_k)
        for _k, _v in data_dict.items():
            self.open[_k] = True
            self._data[_k] = _v


class LockableConfigIndex:
    """
    ConfigParser-like object where sub-dictionaries are LockableConfig's.
    """

    def __init__(self, ini_path):
        self.sections = dict()
        self.update(ini_path)

    def __getitem__(self, key):
        return self.sections[key]

    def update(self, ini_path):
        parser = configparser.ConfigParser()
        parser.read(ini_path)

        for _section in ["DEFAULT", *parser.sections()]:
            _dict = {_k: auto_interpret(_v) for _k, _v in parser[_section].items()}
            if _section in self.sections:
                self.sections[_section].update(_dict)
            else:
                self.sections[_section] = LockableConfig(_dict)
