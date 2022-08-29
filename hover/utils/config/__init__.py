"""
Module for controlling the configuration of hover itself.
"""
import re
import configparser
from typing import Any, Callable, List
from collections import defaultdict


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


class LockableConfigValue:
    """
    Configuration value with additional utilities.
    """

    def __init__(
        self,
        name: str,
        hint: str,
        preprocessor: Callable,
        validation: Callable,
        default: Any,
    ):
        self.name = name
        self.hint = hint
        self.__preprocessor = preprocessor
        self.__validation = validation
        self.__value_lock = False
        self.value = default
        self.example = default

    def parse(self, value):
        value = self.__preprocessor(value)
        assert self.__validation(
            value
        ), f"Validation failed.\nHint: {self.hint}\nExample: {self.example}"
        return value

    @property
    def example(self):
        """
        An example raw (not preprocessed) value.
        """
        return self.__example

    @example.setter
    def example(self, value):
        _ = self.parse(value)
        self.__example = value

    @property
    def value(self):
        """
        A value that, once read, no longer takes assignment.
        """
        self.__value_lock = True
        return self.__value

    @value.setter
    def value(self, value):
        assert not self.locked, f"{self.name} is locked from updates."
        self.__value = self.parse(value)

    @property
    def locked(self):
        return self.__value_lock


class LockableConfig:
    """
    Dict-like object where key reads locks the value from updates.
    """

    def __init__(
        self,
        name: str,
        values: List[LockableConfigValue],
    ):
        self.name = name
        self.__data = dict()
        for _value in values:
            assert isinstance(_value, LockableConfigValue)
            self.__data[_value.name] = _value

    def __getitem__(self, key):
        return self.__data[key].value

    def __setitem__(self, key, value):
        self.__data[key].value = value

    def update(self, data_dict):
        """
        Check all the key lock statuses, then update.
        """
        for _k in data_dict.keys():
            assert not self.__data[_k].locked, f"{_k} is locked from updates."
        for _k, _v in data_dict.items():
            self.__data[_k] = _v

    def hint(self):
        return {
            _k: f"{_v.hint}. Example: {_v.example}" for _k, _v in self.__data.items()
        }

    def items(self):
        return {_k: _v.value for _k, _v in self.__data.items()}


class LockableConfigIndex:
    """
    ConfigParser-like object where sub-dictionaries are LockableConfig's.
    """

    def __init__(
        self,
        configs: List[LockableConfig],
    ):
        self.__configs = dict()
        self.__value_name_to_config_names = defaultdict(list)
        for _config in configs:
            # assign configs
            assert isinstance(_config, LockableConfig)
            self.__configs[_config.name] = _config

            # keep track of value->config name lookup
            for _value_name, _ in _config.items():
                self.__value_name_to_config_name[_value_name].append(_config.name)

    def __getitem__(self, key):
        return self.__configs[key]

    def load_ini(self, ini_path):
        parser = configparser.ConfigParser()
        parser.read(ini_path)

        for _section in parser.sections():
            _dict = {_k: auto_interpret(_v) for _k, _v in parser[_section].items()}
            self.__configs[_section].update(_dict)

    def search(self, value_name):
        return self.__value_name_to_config_name[value_name][:]

    def hint(self):
        return {_k: _v.hint() for _k, _v in self.__configs.items()}
