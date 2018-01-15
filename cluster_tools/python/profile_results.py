#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
############################
Profile CellProfiler Results
############################

## (Preparation for creating the reports)

*Created on Mo, Jan 15 2018, 14:30 by A. Pahl*"""


import argparse
import sys
# import os.path as op
# from collections import Counter

# import pandas as pd

from cellpainting2 import tools as cpt
from cellpainting2 import processing as cpp

cp_config = cpt.load("config")
cp_plates = cpt.load_config("plates")


def usage():
    print("Profile CellProfiler Results\n"
          "Generates the activity profiles and similarity maps.\n"
          "Must be run before creating any reports.\n"
          "profile_results.py -h for more help.")
    sys.exit(1)


def flush_print(txt):
    txt = txt + "\r"
    print(txt)
    sys.stdout.flush()


def profile_plates(plates):
    if plates is not None:
        plates = {plates}
    else:
        plates = cp_plates["Plates"]


if __name__ == "__main__":
    # file_to_search file_w_smiles output_dir job_index
    parser = argparse.ArgumentParser(
        description="Profile the CellProfiler Results.\n"
        "Generates the activity profiles and similarity maps.\n"
        "Must be run before creating any reports.\n"
        "Reads configuration and locations from\n"
        "`config.yaml` and `plates.yaml`.")
    parser.add_argument(
        "-p", "--plate", help="process single plate instead of all data.")

    args = parser.parse_args()

    print("Plate:", args.plate)
    profile_plates(args.plate)
