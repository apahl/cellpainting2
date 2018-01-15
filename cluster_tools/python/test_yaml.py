#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##################
Test Configuration
##################

*Created on Fri Jan 12 2018 12:15 by A. Pahl*"""

import os
import os.path as op
import yaml


def load_config(conf):
    """Loads configuration from default location and
    returns config object.
    Raises error when no configuration could be loaded."""
    if "HOME" in os.environ:
        conf_fn = op.join(os.environ["HOME"], ".config",
                          "cellpainting2", "{}.yaml".format(conf))
    elif "HOMEPATH" in os.environ:  # Windows
        conf_fn = op.join(os.environ["HOMEPATH"],
                          "cellpainting2", "{}.yaml".format(conf))
    try:
        with open(conf_fn, 'r') as ymlfile:
            config = yaml.load(ymlfile)
    except FileNotFoundError:
        print("Configuration file not found.")
        print("Have a look at cellpainting.conf for instructions.")
        raise
    return config


if __name__ == "__main__":
    config = load_config("config")
    print(config["Parameters"][:5])
    print(config["ParameterHelp"])
    print(config["Paths"])
    print("--------------------------------------------------------------------")
    plates = load_config("plates")
    print(plates)
