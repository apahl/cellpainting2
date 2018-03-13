#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#####
Tools
#####

*Created on Thu Jun  7 14:45 2017 by A. Pahl*

Helper Tools acting on individual data..
"""

import os
import os.path as op
import sys
import glob
from collections import Counter, namedtuple
import yaml

import pandas as pd
import numpy as np
import scipy.spatial.distance as dist

from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs


ROWS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
        "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "AB", "AC", "AD", "AE", "AF"]
KEEP = ['Compound_Id', "Batch_Id", "Producer",
        "Address", "Conc_uM", "Smiles", "Pure_Flag"]


def load_config(conf):
    """Loads configuration from default location and
    returns config object.
    Known configuration files are `config.yaml` and `plates.yaml`.
    Raises error when the file could not be loaded."""
    assert conf in ["config", "plates"]

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
        print("Configuration file {}.yaml not found.".format(config))
        print("Have a look at the *.yaml files in the `conf` folder of")
        print("the `cluster_tools directories` for templates and locations.")
        raise
    return config


def is_interactive_ipython():
    try:
        get_ipython()
        ipy = True
        # print("> interactive IPython session.")
    except NameError:
        ipy = False
    return ipy


class Summary(Counter):
    """An OrderedDict-based class that keeps track of the time since its instantiation.
    Used for reporting running details of pipeline functions."""

    def __init__(self, **kwargs):
        """Parameters:
            timeit: whether or not to use the timing functionality. Default: True"""
        super().__init__(**kwargs)

    def __str__(self):
        s_list = []
        keys = sorted(self.keys())
        mlen = max(map(len, keys))
        line_end = "\n"
        for idx, k in enumerate(keys, 1):
            value = self[k]
            s_list.append("{k:{mlen}s}: {val:>7}".format(k=k, mlen=mlen, val=value))
            s_list.append(line_end)

        result = "".join(s_list)
        return result

    def __repr__(self):
        return self.__str__()

    def print(self, final=False):
        keys_len = len(self.keys())
        result = self.__str__()
        if not final:
            result = result + '\033[{}A\r'.format(keys_len)
        print(result, end="")
        sys.stdout.flush()


def profile_sim_dist_corr(current, reference):
    """Calculate the similarity of two activity_profiles of the same length.
    Returns value between 0 .. 1"""

    ref_len = len(reference)
    assert ref_len == len(
        current), "Activity Profiles must have the same length to be compared."
    result = 1 - dist.correlation(current, reference)
    if result < 0.0: result = 0.0
    return result


def profile_sim_tanimoto(p1, p2):
    p_len = len(p1)
    assert p_len == len(p2), "profiles must be of same length!"
    matching = 0
    significant = 0
    for idx in range(p_len):
        if (p1[idx] < 0 and p2[idx] < 0) or (p1[idx] > 0 and p2[idx] > 0):
            matching += 1
        if p1[idx] != 0.0 or p2[idx] != 0.0:
            significant += 1
    result = matching / significant
    return result


def subtract_profiles(prof1, prof2):
    """Subtract prof2 from prof1. A new profile is returned."""
    prof1_len = len(prof1)
    assert prof1_len == len(prof2), "Activity Profiles must have the same length to be compared."
    result = []
    for idx in range(prof1_len):
        d = prof1[idx] - prof2[idx]
        if abs(d) <= 1.58: d = 0.0
        result.append(d)
    return result


def del_nz_positions(prof1, prof2):
    """Set positions that are non-zero in both profiles to zero. A new profile is returned."""
    prof1_len = len(prof1)
    assert prof1_len == len(prof2), "Activity Profiles must have the same length to be compared."
    result = []
    for idx in range(prof1_len):
        if prof1[idx] != 0.0 and prof2[idx] != 0.0:
            result.append(0.0)
        else:
            result.append(prof1[idx])
    return result


def mol_from_smiles(smi):
    if not isinstance(smi, str):
        smi = "*"
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        mol = Chem.MolFromSmiles("*")
    return mol


def chem_sim(mol_fp, query_smi):
    query = mol_from_smiles(query_smi)
    if len(query.GetAtoms()) > 1:
        query_fp = Chem.GetMorganFingerprint(query, 2)  # ECFC4
        return round(DataStructs.TanimotoSimilarity(mol_fp, query_fp), 3)
    return np.nan


def split_plate_name(full_name, sep="-"):
    """Split the full platename into (date, plate).
    Returns a namedtuple or None, if the name spec is not met."""
    parts = full_name.split(sep=sep, maxsplit=1)
    if len(parts) == 0:
        return None  # The full plate name needs to contain at least one '-'.
    if len(parts[0]) != 6:
        return None  # The date has to be of format <yymmdd>.
    Plate = namedtuple("Plate", ["date", "name"])
    result = Plate(date=parts[0], name=parts[1])
    return result


def get_plates_in_dir(dir, exclude=["layout"]):
    """Return a list of all plates in the given dir.
    Performs a search of the subdirs in the dir and adds plate if it conforms
    to the `split_plate_name` spec.
    Returns a list of full plate name strings."""
    result = []
    plate_dir = op.join(dir, "*")
    for dir_name in glob.glob(plate_dir):
        skip = False
        for x in exclude:
            if x in dir_name:
                skip = True
                break
        if skip: continue
        if not op.isdir(dir_name): continue
        plate_name = op.split(dir_name)[1]
        plate = split_plate_name(plate_name)
        if plate is None: continue  # the dir_name does not conform to the spec
        result.append(plate_name)  # apped the full platename as string
    return result


def format_well(well):
    """Fix well format, e.g. `A1` --> `A01`."""
    wl = len(well)
    assert wl >= 2 and wl <= 4, "well has to have 2 - 4 characters!"
    column = []
    row = []
    for pos in range(wl):
        c = well[pos]
        if c.isalpha():
            row.append(c.upper())
            continue
        row_str = "".join(row)
        assert row_str in ROWS, "row {} is not a valid row.".format(row_str)
        column.append(c)
    if len(column) < 2:
        column.insert(0, "0")  # prepend a zero
    result = row
    result.extend(column)
    return "".join(result)


def well_from_position_single(row, col):
    result = [ROWS[row - 1], "{:02d}".format(col)]
    return "".join(result)


def position_from_well_single(well):
    wl = len(well)
    column = []
    row = []
    for pos in range(wl):
        c = well[pos]
        if c.isalpha():
            row.append(c.upper())
            continue
        row_str = "".join(row)
        try:
            row_num = ROWS.index(row_str) + 1
        except ValueError:
            raise ValueError("row {} is not a valid row.".format(row_str))
        column.append(c)
    column_num = int("".join(column))
    return row_num, column_num


def find_dups(it):
    """Find duplicates in an iterable."""
    ctr = Counter(it)
    result = {}
    for c in ctr:
        if ctr[c] > 1:
            result[c] = ctr[c]
    return result


def diff(it1, it2):
    """Find the differences between two iterables"""
    s2 = set(it2)
    diff = [x for x in it1 if x not in s2]
    return diff


def print_dir(obj):
    for f in dir(obj):
        if not f.startswith("_"):
            print(f)


def print_iter(l):
    for x in l:
        print(x)


def create_dirs(path):
    if not op.isdir(path):
        os.makedirs(path)


def empty_dir(path):
    """Remove all file in the given directory."""
    if not op.isdir(path): return
    for fn in os.listdir(path):
        full_name = op.join(path, fn)
        if op.isfile(full_name):
            os.unlink(full_name)


def middle(lst, size):
    """Return the middle fraction of a sorted list, removing outliers."""
    mid_lst = sorted(list(lst))
    l_mid_lst = len(mid_lst)
    num_el = int(size * l_mid_lst)
    start = (l_mid_lst - num_el) // 2
    end = start + num_el
    mid_lst = mid_lst[start:end]
    return mid_lst


def melt(df, id_prop="Compound_Id"):
    """Taken and adapted from the Holoviews measles heatmap example."""
    result = pd.melt(df, id_vars=id_prop,
                     var_name="Parameter", value_name="Value")
    result = result.reset_index().drop("index", axis=1)
    # result = result.sort_values([id_prop, "Parameter"]).reset_index().drop("index", axis=1)
    result = result[["Parameter", id_prop, "Value"]]
    return result


def show_available_plates():
    config = load_config("config")
    plates = get_plates_in_dir(config["Dirs"]["PlatesDir"])
    print("Available Plates:")
    for plate_full_name in plates:
        print("  -", plate_full_name)
