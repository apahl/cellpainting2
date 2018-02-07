#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
############################
Profile CellProfiler Results
############################

## (Preparation for creating the reports)
*Created on Mo, Jan 15 2018, 14:30 by A. Pahl*

Plate names need to follow this scheme:
    date-platename
    where the date should be in the form: YYMMDD
    and the platename can optionally contain a quadrant (but does not have to):
        S0203-1 or C2017-1
"""


import argparse
import sys
import os.path as op
# from collections import Counter

# import pandas as pd

from cellpainting2 import tools as cpt
from cellpainting2 import processing as cpp

cp_config = cpt.load_config("config")


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


def profile_plates(plates=None, tasks=None):
    """`plate` has to be of the form: <yymmdd>-<platename>-<quadrant>,
    e.g.: `180123-S0195-1` or `180123-C2017-1`.
    `plates` can be a string, a comma-separated string of names, a list of strings,
    or None (then all plates present in the directory given in the config
    will be processed)."""
    if plates is None:
        plates = cpt.get_plates_in_dir(cp_config["Dirs"]["PlatesDir"])
    elif isinstance(plates, str):
        plates = plates.split(",")
    print("\nThe following plates will be processed:")
    print(plates)

    if tasks is None:
        print("\nPHASE 1: Processing Data...")
        for plate_full_name in plates:
            plate = cpt.split_plate_name(plate_full_name)  # a namedtuple
            if plate is None:
                raise ValueError("Plate {} does not follow the spec.".format(plate_full_name))
            flush_print("\nProcessing plate {}-{} ...".format(plate.date, plate.name))
            src_templ = op.join(cp_config["Dirs"]["PlatesDir"], "{}-{}")
            src_dir = src_templ.format(plate.date, plate.name)
            #                                                   process as Pandas
            ds_plate = cpp.read_csv(op.join(src_dir, "Results.tsv")).compute()
            ds_plate = ds_plate.group_on_well()
            ds_plate.position_from_well()  # inplace
            ds_plate = ds_plate.remove_skipped_echo_direct_transfer(op.join(src_dir, "*_print.xml"))
            ds_plate.well_type_from_position()

            ds_plate.flag_toxic()
            # ds_plate.write_csv(op.join(cp_config["Dirs"]["DataDir"], "tmp_toxic.tsv"),
            #                    parameters=cpp.FINAL_PARAMETERS)
            ds_plate = ds_plate.activity_profile(act_cutoff=1.58)
            ds_plate = ds_plate.join_layout_1536(plate.name)
            ds_plate.data["Plate"] = "{}-{}".format(plate.date, plate.name)

            ds_plate = ds_plate.join_smiles()
            ds_plate = ds_plate.join_batch_data()
            # ds_plate.write_csv(op.join(cp_config["Dirs"]["DataDir"], "tmp_batch.tsv"),
            #                    parameters=cpp.FINAL_PARAMETERS)
            ds_plate.update_datastore(write=True)

        print("\nPHASE 2: Extracting References...")
        cpp.extract_references()

    if tasks is None or tasks == "sim":
        print("\nPHASE 3: Finding Similar References...")
        cpp.update_similar_refs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile the CellProfiler Results.\n"
                    "Generates the activity profiles.\n"
                    "Must be run before creating any reports.\n"
                    "Reads configuration from `config.yaml`.\n")
    parser.add_argument("-p", "--plate",
                        help="Process single plates instead of all data. "
                             "Provide single plate name or comma-separated list.")
    parser.add_argument("-t", "--tasks", choices=["sim"],
                        help="Perform only the specified action.")

    args = parser.parse_args()

    print("Plate:", args.plate)
    print("Tasks:", args.tasks)
    profile_plates(plates=args.plate, tasks=args.tasks)
