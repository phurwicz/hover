"""
Module root where constants get configured.
"""
import os
from hover.utils.config import LockableConfigIndex

config = LockableConfigIndex(
    os.path.join(
        os.path.dirname(__file__),
        "config.ini",
    )
)
