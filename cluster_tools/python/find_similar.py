#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#######################
Find Similar References
#######################

Version for parallel use on the cluster.

*Created on Thu Aug 22 12:00 2017 by A. Pahl*"""


import sys
import argparse
# import os.path as op


from cellpainting2 import tools as cpt
from cellpainting2 import processing as cpp

cp_config = cpt.load_config("config")


def find_similar(plates, taskid):
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
    cpp.load_resource("DATASTORE")
    ds = cpp.DATASTORE[cpp.DATASTORE["Plate"].isin(plates)].compute()
    # inparallel defers the times_found determination
    cpp.update_similar_refs(ds, inparallel=True, taskid=taskid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find similar references.\n"
                    "For parallel use on the cluster.\n"
                    "Reads configuration from `config.yaml`.")
    parser.add_argument("-p", "--plates",
                        help="Process single plates instead of all data. "
                             "Provide single plate name or comma-separated list.")
    parser.add_argument("-t", "--taskid", type=int,
                        help="Slurm array task Id. Requires `plates` to be given "
                             "and `ntasks` to be also set.")
    parser.add_argument("-n", "--ntasks", type=int,
                        help="Number of Slurm array tasks in total.")

    args = parser.parse_args()

    print("Plates:    ", args.plates)

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

    find_similar(plates=args.plates, taskid=args.taskid)
