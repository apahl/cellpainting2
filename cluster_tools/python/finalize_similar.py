#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###########################
Finalize Similar References
###########################

Finalizing Similar References after parallel on cluster.
Collects the times found.

*Created on Thu Aug 22 12:00 2017 by A. Pahl*"""


import argparse
import os.path as op

from cellpainting2 import tools as cpt
from cellpainting2 import processing as cpp

cp_config = cpt.load_config("config")


def finalize_similar():
    """`plate` has to be of the form: <yymmdd>-<platename>-<quadrant>,
    e.g.: `180123-S0195-1` or `180123-C2017-1`.
    `plates` can be a string, a comma-separated string of names, a list of strings,
    or None (then reports for all plates present in the directory given in the config
    will be generated)."""
    tmp_dir = op.join(cp_config["Dirs"]["DataDir"], "tmp")
    tmp_file = op.join(tmp_dir, "sim_tmp-*.tsv")
    cpp.sim_times_found(tmp_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finalize similar references.\n"
                    "Has to be run after the parallel run of find_similar.\n"
                    "Has to be run as a single job.\n"
                    "Reads configuration from `config.yaml`.")

    args = parser.parse_args()

    finalize_similar()
