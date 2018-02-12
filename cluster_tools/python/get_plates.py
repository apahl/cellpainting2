#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##########
Get Plates
##########

*Created on Mon, Feb 12 2018 by A. Pahl*"""

from cellpainting2 import tools as cpt

cp_config = cpt.load_config("config")


def get_plates():
    """Return the list of avaiable plates (for use in cluster array tasks)."""
    plates = cpt.get_plates_in_dir(cp_config["Dirs"]["PlatesDir"])
    plates.sort()
    print(",".join(plates))


if __name__ == "__main__":
    get_plates()
