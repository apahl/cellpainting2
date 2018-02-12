#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##############################
Aggregate CellProfiler Results
##############################

*Created on Thu Aug 22 12:00 2017 by A. Pahl*"""


import sys
import argparse
import os.path as op
# from collections import Counter

# import pandas as pd

from cellpainting2 import tools as cpt
from cellpainting2 import processing as cpp, reporting as cpr

cp_config = cpt.load_config("config")


def create_reports(plates=None):
    """`plate` has to be of the form: <yymmdd>-<platename>-<quadrant>,
    e.g.: `180123-S0195-1` or `180123-C2017-1`.
    `plates` can be a string, a comma-separated string of names, a list of strings,
    or None (then reports for all plates present in the directory given in the config
    will be generated)."""
    if plates is None:
        plates = cpt.get_plates_in_dir(cp_config["Dirs"]["PlatesDir"])
    elif isinstance(plates, str):
        plates = plates.split(",")
    print("\nThe following plates will be processed:")
    print(plates)
    cpp.load_resource("DATASTORE")
    ds = cpp.DATASTORE
    for plate_full_name in plates:
        plate = cpt.split_plate_name(plate_full_name)  # a namedtuple
        if plate is None:
            raise ValueError("Plate {} does not follow the spec.".format(plate_full_name))
        print("\nCreating report for plate {}-{} ...".format(plate.date, plate.name))
        src_templ = op.join(cp_config["Dirs"]["PlatesDir"], "{}-{}")
        src_dir = src_templ.format(plate.date, plate.name)

        ds_profile = ds[ds["Plate"] == plate_full_name].compute()
        ds_profile = ds_profile.sort_values(["Toxic", "Activity"],
                                            ascending=[True, False])
        cpr.full_report(ds_profile, src_dir, report_name=plate_full_name,
                        plate=plate_full_name, highlight=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Cell Painting Reports.\n"
                    "Writes the reports into the dir specified in config.\n"
                    "Can be used after profile_plates.py has been run.\n"
                    "Reads configuration from `config.yaml`.")
    parser.add_argument("-p", "--plates",
                        help="Process single plates instead of all data. "
                             "Provide single plate name or comma-separated list.")
    parser.add_argument("-s", "--showavail", action="store_true",
                        help="Show available plates.")
    parser.add_argument("-t", "--taskid", type=int,
                        help="Slurm array task Id. Requires `plates` to be given "
                             "and `ntasks` to be also set.")
    parser.add_argument("-n", "--ntasks", type=int,
                        help="Number of Slurm array tasks in total.")

    args = parser.parse_args()

    print("Plates:    ", args.plates)

    if args.showavail:
        cpt.show_available_plates()
        sys.exit(0)

    if args.taskid is not None:
        if args.ntasks is None:
            print("ntasks needs to be given.")
            sys.exit(1)
        if args.plates is None:
            print("plates needs to be given.")
            sys.exit(1)

        taskid = args.taskid
        ntasks = args.ntasks
        plates = args.plates
        plates = plates.split(",")
        nplates = len(plates)
        lbound = int((taskid - 1) / ntasks * nplates)
        ubound = int(taskid / ntasks * nplates)
        plates_slice = plates[lbound:ubound]
        args.plates = ",".join(plates_slice)

    create_reports(plates=args.plates)
