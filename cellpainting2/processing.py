#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##########
Processing
##########

*Created on Thu Jun  1 14:15 2017 by A. Pahl*

Processing results from the CellPainting Assay in standalone progs
or interactively in the Jupyter notebook.
This module provides the DataSet class and its methods.
Additional functions in this module act on Dask or Pandas DataFrames."""

import sys
import time
import glob
import math
import os.path as op
# from collections import Counter
import xml.etree.ElementTree as ET
import pickle

import pandas as pd
import numpy as np
from dask import dataframe as dd

from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs

from cellpainting2 import tools as cpt
cp_config = cpt.load_config("config")
IPYTHON = cpt.is_interactive_ipython()
if IPYTHON:
    from IPython.core.display import HTML

ACT_PROF_PARAMETERS = cp_config["Parameters"]

LIMIT_SIMILARITY_L = cp_config["Cutoffs"]["LimitSimilarityL"]
LIMIT_CELL_COUNT_L = cp_config["Cutoffs"]["LimitCellCountL"]
LIMIT_ACTIVITY_L = cp_config["Cutoffs"]["LimitActivityL"]
INPLACE = cp_config["Options"]["InPlace"]
CHUNKSIZE = int(cp_config["Options"]["ChunkSize"])
NPARTITIONS = int(cp_config["Options"]["NPartitions"])

try:
    from misc_tools import apl_tools
    AP_TOOLS = True
    print("* Cell Painting v2")
    #: Library version
    VERSION = apl_tools.get_commit(__file__)
    # I use this to keep track of the library versions I use in my project notebooks
    print("{:45s} (commit: {})".format(__name__, VERSION))

except ImportError:
    AP_TOOLS = False
    print("{:45s} ({})".format(__name__, time.strftime(
        "%y%m%d-%H:%M", time.localtime(op.getmtime(__file__)))))

FINAL_PARAMETERS = ['Metadata_Plate', 'Metadata_Well', 'plateColumn', 'plateRow',
                    "Compound_Id", 'Container_Id', "Well_Id", "Producer", "Pure_Flag",
                    "Toxic", "Is_Ref",
                    "Rel_Cell_Count", "Known_Act", "Trivial_Name", 'WellType', 'Conc_uM',
                    "Activity", "Plate", "Smiles"]
DROP_FROM_NUMBERS = ['plateColumn', 'plateRow', 'Conc_uM', "Compound_Id"]
DROP_GLOBAL = {"PathName_CellOutlines", "URL_CellOutlines", 'FileName_CellOutlines',
               'ImageNumber', 'Metadata_Site', 'Metadata_Site_1', 'Metadata_Site_2'}  # set literal
QUANT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEBUG = False


def debug_print(txt, val):
    if DEBUG:
        txt = txt + ":"
        print("DEBUG   {:20s}".format(txt), val)


def is_pandas(df):
    return isinstance(df, pd.DataFrame)


def is_dask(df):
    return isinstance(df, dd.DataFrame)


class DataSet():
    def __init__(self, log=True):
        self.data = None
        self.fields = {"plateColumn": "Metadata_Plate",
                       "WellType": "WellType", "ControlWell": "Control", "CompoundWell": "Compound"}
        self.log = log

    def __getitem__(self, key):
        res = self.data[key]
        if is_pandas(res) or is_dask(res):
            result = self.new()
            result.data = res
            print_log(result.data, "subset")
        else:
            result = res
        return result

    def __setitem__(self, key, item):
        self.data[key] = item

    def __getattr__(self, name):
        """Try to call undefined methods on the underlying pandas DataFrame."""
        if hasattr(self.data, name):
            def method(*args, **kwargs):
                res = getattr(self.data, name)(*args, **kwargs)
                if isinstance(res, pd.DataFrame):
                    result = self.new()
                    result.data = res
                    print_log(result.data, name)
                else:
                    result = res
                return result
            return method
        else:
            raise AttributeError

    def _repr_html_(self):
        result = drop_cols(self.data, FINAL_PARAMETERS)
        return result.to_html()

    def new(self):
        result = DataSet()
        return result

    def copy(self):
        result = self.new()
        result.data = self.data.copy()
        return result

    @property
    def is_pandas(self):
        return isinstance(self.data, pd.DataFrame)

    @property
    def is_dask(self):
        return isinstance(self.data, dd.DataFrame)

    @property
    def data_type(self):
        """Returns the underlying data structure as string."""
        if self.is_pandas:
            dtype = "pandas"
        elif self.is_dask:
            dtype = "dask"
        else:
            dtype = "unknown"
        return dtype

    def show(self):
        parameters = [k for k in FINAL_PARAMETERS if k in self.data]
        print("Shape:     ", self.shape)
        print("Parameters:", parameters)
        if IPYTHON:
            return HTML(self.data[parameters]._repr_html_())
        else:
            print(self.data[parameters].__repr__())

    def head(self, n=5):
        parameters = [k for k in FINAL_PARAMETERS if k in self.data]
        res = self.data[parameters].head(n)
        result = DataSet()
        result.data = res
        result.print_log("head")
        return result

    def keys(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data.keys()
        else:  # Dask DataFrames have no `keys()` method
            return self.data.columns

    def compute(self):
        if is_dask(self.data):
            df = self.data.compute()
        else:
            df = self.data.copy()
        result = self.new()
        result.data = df
        print_log(df, "compute")
        return result

    def drop_cols(self, cols, inplace=False):
        """Drops the list of columns from the DataFrame.
        Listed columns that are not present in the DataFrame are simply ignored
        (no error is thrown)."""
        if inplace:
            self.data = drop_cols(self.data, cols, info="(inplace)")
        else:
            result = DataSet()
            result.data = drop_cols(self.data, cols)
            result.print_log("drop cols")
            return result

    def keep_cols(self, cols, inplace=False):
        drop = list(set(self.data.columns) - set(cols))
        if inplace:
            self.data = self.data.drop(drop, axis=1)
        else:
            result = DataSet()
            result.data = self.data.drop(drop, axis=1)
            result.print_log("keep cols")
            return result

    def print_log(self, component, add_info=""):
        if self.log:
            print_log(self.data, component, add_info)

    def read_csv(self, fn, sep="\t"):
        """Read one or multiple result files and concatenate them into one dataset,
        containing a Dask DataFrame data object.
        `fn` is a single filename (string) or a list of filenames."""
        self.data = read_csv(fn, sep=sep).data

    def write_csv(self, fn, parameters=None, sep="\t"):
        if parameters is None:
            parameters = list(self.data.columns)
        result = self.keep_cols(parameters)
        if self.is_dask():
            result.data.to_csv(fn, sep=sep, index=False, chunksize=CHUNKSIZE)
        else:
            result.data.to_csv(fn, sep=sep, index=False)
        result.print_log("write csv")

    def write_pkl(self, fn):
        """`data` has to be a Pandas DataFrame."""
        assert self.is_pandas, "`data` has to be a Pandas DataFrame."
        self.data.to_pickle(fn)
        self.print_log("write pkl")

    def write_parameters(self, fn="parameters.txt"):
        parameters = sorted(self.measurements)
        with open("parameters.txt", "w") as f:
            f.write('"')
            f.write('",\n"'.join(parameters))
            f.write('"')
        print(len(parameters), "parameters written.")

    def info(self, final_only=True):
        """Show a summary of the MolFrame."""
        if final_only:
            keys = list(set(FINAL_PARAMETERS).intersection(
                set(self.data.columns)))
        else:
            keys = list(self.data.columns)
        if is_dask(self.data):
            print("Dask DataFrame, Columns:", keys)
        else:
            info = []
            for k in keys:
                info.append({"Field": k, "Count": self.data[k].notna(
                ).count(), "Type": str(self.data[k].dtype)})
            info.append({"Field": "Total", "Type": "",
                         "Count": self.data.shape[0]})
            return pd.DataFrame(info)

    def describe(self, times_mad=3.0):
        df = numeric_parameters(self.data)
        stats = pd.DataFrame()
        stats["Min"] = df.min()
        stats["Max"] = df.max()
        stats["Median"] = df.median()
        stats["MAD"] = df.mad()
        stats["Outliers"] = df[(
            ((df - df.median()).abs() - times_mad * df.mad()) > 0)].count()
        print(self.shape)
        return stats

    def well_type_from_position(self):
        """Assign the WellType from the position on the plate.
        Controls are in column 11 and 12
        Operates inplace!"""
        well_type_from_position(self.data)

    def well_from_position(self, well_name="Metadata_Well",
                           row_name="plateRow", col_name="plateColumn"):
        """Assign Metadata_Well from plateRow, plateColumn
        Operates inplace!"""
        well_from_position(self.data, well_name=well_name,
                           row_name=row_name, col_name=col_name)

    def position_from_well(self, well_name="Metadata_Well",
                           row_name="plateRow", col_name="plateColumn"):
        """Generate plateRow and plateColumn from Metatadata_Well
        Operates inplace!"""
        position_from_well(self.data, well_name=well_name,
                           row_name=row_name, col_name=col_name)

    def join_layout_1536(self, plate, quadrant="", how="inner"):
        """Cell Painting is always run in 384er plates.
        COMAS standard screening plates are format 1536.
        With this function, the 1536-to-384 reformatting file
        with the smiles added by join_smiles_to_layout_1536()
        can be used directly to join the layout to the individual 384er plates.
        `data` has to be a Pandas DataFrame."""
        assert self.is_pandas, "`data` has to be a Pandas DataFrame."
        result = DataSet(log=self.log)
        result.data = join_layout_1536(
            self.data, plate, quadrant)
        return result

    def numeric_parameters(self):
        result = DataSet()
        result.data = numeric_parameters(self.data)
        return result

    def flag_toxic(self, cutoff=LIMIT_CELL_COUNT_L / 100):
        """Flag data rows of toxic compounds
        Operates inplace!"""
        flag_toxic(self.data, cutoff=cutoff)

    def remove_toxic(self, cutoff=LIMIT_CELL_COUNT_L / 100):
        """Remove data rows of toxic compounds"""
        result = DataSet()
        toxic = DataSet()
        result.data, toxic.data = remove_toxic(self.data, cutoff=cutoff)
        return result, toxic

    def remove_impure(self, strict=False, reset_index=True):
        """Remove entries with `Pure_Flag == "Fail"`"""
        result = DataSet()
        flagged = DataSet()
        result.data, flagged.data = remove_impure(self.data)
        result.print_log(
            "remove impure", "{:3d} removed".format(flagged.shape[0]))
        return result, flagged

    def remove_skipped_echo_direct_transfer(self, fn):
        """Remove wells that were reported as skipped in the Echo protocol (xml).
        This functions works with Echo direct transfer protocols.
        Function supports using wildcards in the filename, the first file will be used.
        Returns a new dataframe without the skipped wells."""
        result = DataSet()
        result.data, skipped = remove_skipped_echo_direct_transfer(self.data, fn=fn)
        return result

    def drop_dups(self, cpd_id="Compound_Id"):
        """Drop duplicate Compound_Ids"""
        result = DataSet()
        result.data = self.data.drop_duplicates(subset=[cpd_id])
        result.print_log("drop dups")
        return result

    def group_on_well(self, group_by=FINAL_PARAMETERS):
        """Group results on well level."""
        result = DataSet()
        result.data = group_on_well(self.data, group_by=group_by)
        return result

    def join_batch_data(self, df_data=None, how="left", fillna="n.d."):
        """Join data by Batch_Id.
        `data` has to be a Pandas DataFrame."""
        assert self.is_pandas, "`data` has to be a Pandas DataFrame."
        result = DataSet()
        result.data = join_batch_data(
            self.data, df_data=df_data, how=how, fillna=fillna)
        return result

    def join_container_data(self, df_data=None, how="left", fillna=""):
        """Join data by Container_Id.
        `data` has to be a Pandas DataFrame."""
        assert self.is_pandas, "`data` has to be a Pandas DataFrame."
        result = DataSet()
        result.data = join_container_data(
            self.data, df_data=df_data, how=how, fillna=fillna)
        return result

    def join_container(self, cont_data=None, how="inner"):
        """`data` has to be a Pandas DataFrame."""
        assert self.is_pandas, "`data` has to be a Pandas DataFrame."
        result = DataSet(log=self.log)
        result.data = join_container(self.data, cont_data=cont_data, how=how)
        return result

    def join_smiles(self, df_smiles=None, how="left"):
        """Join Smiles from Compound_Id.
        `data` has to be a Pandas DataFrame."""
        assert self.is_pandas, "`data` has to be a Pandas DataFrame."
        result = DataSet()
        result.data = join_smiles(self.data, df_smiles=df_smiles, how=how)
        return result

    def join_annotations(self):
        """Join Annotations from Compound_Id.`data` has to be a Pandas DataFrame."""
        assert self.is_pandas, "`data` has to be a Pandas DataFrame."
        result = DataSet()
        result.data = join_annotations(self.data)
        return result


    def activity_profile(self, parameters=ACT_PROF_PARAMETERS, act_cutoff=1.58, only_final=True):
        """Calculates the log2-fold values for the parameters.

        If a list of parameters is given, then the activity profile will be
        calculated for these parameters.

        If `only_final` == `True`, then only the parameters listed in `FINAL_PARAMETERS`
        are kept in the output_table.
        `data` has to be a Pandas DataFrame.

        Returns a new DataSet."""
        assert self.is_pandas, "`data` has to be a Pandas DataFrame."
        result = DataSet()
        result.data = activity_profile(self.data, parameters=parameters,
                                       act_cutoff=act_cutoff, only_final=only_final)
        return result

    def relevant_parameters(self, ctrls_std_rel_min=0.001,
                            ctrls_std_rel_max=0.10):
        """`data` has to be a Pandas DataFrame."""
        assert self.is_pandas, "`data` has to be a Pandas DataFrame."
        result = DataSet()
        result.data = relevant_parameters(self.data, ctrls_std_rel_min=ctrls_std_rel_min,
                                          ctrls_std_rel_max=ctrls_std_rel_max)
        return result

    def id_filter(self, cpd_ids, id_col="Compound_Id", reset_index=True,
                  sort_by_input=False):
        result = self.new()
        result.data = id_filter(self.data, cpd_ids, id_col=id_col, reset_index=reset_index,
                                sort_by_input=sort_by_input)
        return result

    def update_similar_refs(self, write=True):
        """Find similar compounds in references and update the export file.
        The export file of the dict object is in tsv format. In addition,
        a tsv file with only the most similar reference is written for use in PPilot.
        This method does not return anything, it just writes the result to file."""
        update_similar_refs(self.data, write=write)

    def update_datastore(self):
        """Update the DataStore with the current DataFrame."""
        update_datastore(self.data)

    def find_similar(self, act_profile, cutoff=0.5, max_num=5):
        """Filter the dataframe for activity profiles similar to the given one.
        `cutoff` gives the similarity threshold, default is 0.5."""
        result = DataSet()
        result.data = find_similar(
            self.data, act_profile=act_profile, cutoff=cutoff, max_num=max_num)
        result.print_log("find similar")
        return result

    def well_id_similarity(self, well_id1, well_id2):
        """Calculate the similarity of the activity profiles from two compounds
        (identified by `Compound_Id`). Returns value between 0 .. 1"""
        return well_id_similarity(self.data, well_id1, self.data, well_id2)

    @property
    def shape(self):
        return self.data.shape

    @property
    def metadata(self):
        """Returns a list of the those parameters in the DataFrame that are NOT CellProfiler measurements."""
        return metadata(self.data)

    @property
    def measurements(self):
        """Returns a list of the CellProfiler parameters that are in the DataFrame."""
        return measurements(self.data)


def read_csv(fn, sep="\t"):
    """Read one or multiple result files and concatenate them into one dataset,
    containing a Dask DataFrame data object.
    `fn` is a single filename (string) or a list of filenames."""

    if isinstance(fn, list):
        df = dd.concat((pd.read_csv(f, sep=sep) for f in fn))
    else:
        df = dd.read_csv(fn, sep=sep)
    cols = set(df.columns)
    drop = DROP_GLOBAL.intersection(cols)
    df.drop(drop, axis=1)
    result = DataSet()
    result.data = df
    result.print_log("read dataset")
    return result


def read_pkl(fn):
    result = DataSet()
    result.data = pd.read_pickle(fn)
    result.print_log("load pickle")
    return result


def flush_print(txt, end=""):
    txt = txt + "\r"
    print(txt, end=end)
    sys.stdout.flush()


def print_log(df, component, add_info=""):
    component = component + ":"
    if len(add_info) > 0:
        add_info = "    ({})".format(add_info)
    if is_pandas(df):
        print("* {:22s} ({:5d} | {:4d}){}".format(component,
              df.shape[0], df.shape[1], add_info))
    elif is_dask(df):
        df_cols = len(df.columns)
        print("* {:22s} ( dask | {:4d}){}".format(component,
              df_cols, add_info))
    else:
        print("* {:22s} ( unknown    )".format(component))


def clear_resources():
    try:
        del SMILES
        print("* deleted resource: SMILES")
    except NameError:
        pass
    try:
        del ANNOTATIONS
        print("* deleted resource: ANNOTATIONS")
    except NameError:
        pass
    try:
        del REFERENCES
        print("* deleted resource: REFERENCES")
    except NameError:
        pass
    try:
        del SIM_REFS
        print("* deleted resource: SIM_REFS")
    except NameError:
        pass
    try:
        del DATASTORE
        print("* deleted resource: DATASTORE")
    except NameError:
        pass
    try:
        del LAYOUTS
        print("* deleted resource: LAYOUTS")
    except NameError:
        pass


def read_resource(res, mode="cpd"):
    res = res.lower()
    if "sim" in res:
        print("  - reading resource:                      (SIM_REFS)")
        if "ext" in mode.lower():
            srp = cp_config["Paths"]["SimRefsExtPath"]
        else:
            srp = cp_config["Paths"]["SimRefsPath"]
        try:
            result = dd.read_csv(srp, sep="\t")
        except (FileNotFoundError, OSError):
            print("  * SIM_REFS not found, creating new one.")
            result = pd.DataFrame()
    elif "datast" in res:
        print("  - reading resource:                      (DATASTORE)")
        try:
            result = dd.read_csv(cp_config["Paths"]["DatastorePath"], sep="\t")
        except (FileNotFoundError, OSError):
            print("  * DATASTORE not found, creating new one.")
            result = pd.DataFrame()
    else:
        print("  * Resource {} not found, creating new one.".format(res.upper()))
        result = pd.DataFrame()
    return result


def load_resource(resource, force=False, mode="cpd", limit_cols=True):
    """Available resources: SMILES, ANNOTATIONS, SIM_REFS, REFERENCES,
                            CONTAINER, CONTAINER_DATA, BATCH_DATA, DATASTORE, LAYOUTS"""
    res = resource.lower()
    glbls = globals()
    if "smi" in res:
        if force or "SMILES" not in glbls:
            # except NameError:
            global SMILES
            print("  - loading resource:                      (SMILES)")
            SMILES = dd.read_csv(cp_config["Paths"]["SmilesPath"], sep="\t")
            if isinstance(limit_cols, list):
                SMILES = SMILES[limit_cols]
            elif limit_cols is True and len(cp_config["Paths"]["SmilesCols"]) > 0:
                SMILES = SMILES[cp_config["Paths"]["SmilesCols"]]
            # SMILES = SMILES.apply(pd.to_numeric, errors='ignore', axis=1)
    elif "anno" in res:
        if force or "ANNOTATIONS" not in glbls:
            global ANNOTATIONS
            print("  - loading resource:                      (ANNOTATIONS)")
            ANNOTATIONS = dd.read_csv(
                cp_config["Paths"]["AnnotationsPath"], sep="\t")
    elif "sim" in res:
        if force or "SIM_REFS" not in glbls:
            global SIM_REFS
            print("  - loading resource:                      (SIM_REFS)")
            if "ext" in mode.lower():
                srp = cp_config["Paths"]["SimRefsExtPath"]
            else:
                srp = cp_config["Paths"]["SimRefsPath"]
            try:
                SIM_REFS = dd.read_csv(srp, sep="\t")
            except (FileNotFoundError, OSError):
                print("  * SIM_REFS not found, creating new one.")
                SIM_REFS = pd.DataFrame()
    elif "ref" in res:
        if force or "REFERENCES" not in glbls:
            global REFERENCES
            print("  - loading resource:                      (REFERENCES)")
            REFERENCES = dd.read_csv(
                cp_config["Paths"]["ReferencesPath"], sep="\t")
    elif "cont" in res:
        if force or "CONTAINER" not in glbls:
            global CONTAINER
            print("  - loading resource:                      (CONTAINER)")
            CONTAINER = dd.read_csv(
                cp_config["Paths"]["ContainerPath"], sep="\t")
            if isinstance(limit_cols, list):
                CONTAINER = CONTAINER[limit_cols]
            elif limit_cols is True and len(cp_config["Paths"]["ContainerCols"]) > 0:
                CONTAINER = CONTAINER[cp_config["Paths"]["ContainerCols"]]
    elif "container_d" in res:
        if force or "CONTAINER_DATA" not in glbls:
            global CONTAINER_DATA
            print("  - loading resource:                      (CONTAINER)")
            CONTAINER_DATA = dd.read_csv(
                cp_config["Paths"]["ContainerDataPath"], sep="\t")
            if isinstance(limit_cols, list):
                CONTAINER_DATA = CONTAINER_DATA[limit_cols]
            elif limit_cols is True and len(cp_config["Paths"]["ContainerDataCols"]) > 0:
                CONTAINER_DATA = CONTAINER_DATA[cp_config["Paths"]
                                                ["ContainerDataCols"]]
    elif "batch_d" in res:
        if force or "BATCH_DATA" not in glbls:
            global BATCH_DATA
            print("  - loading resource:                      (BATCH_DATA)")
            BATCH_DATA = dd.read_csv(
                cp_config["Paths"]["BatchDataPath"], sep="\t",
                dtype={"Identity": np.object, "LCMS_Date": np.object})
            if isinstance(limit_cols, list):
                BATCH_DATA = BATCH_DATA[limit_cols]
            elif limit_cols is True and len(cp_config["Paths"]["BatchDataCols"]) > 0:
                BATCH_DATA = BATCH_DATA[cp_config["Paths"]["BatchDataCols"]]
    elif "datast" in res:
        if force or "DATASTORE" not in glbls:
            global DATASTORE
            print("  - loading resource:                      (DATASTORE)")
            try:
                DATASTORE = dd.read_csv(
                    cp_config["Paths"]["DatastorePath"], sep="\t")
            except (FileNotFoundError, OSError):
                print("  * DATASTORE not found, creating new one.")
                DATASTORE = pd.DataFrame()
    elif "layout" in res:
        if force or "LAYOUTS" not in glbls:
            global LAYOUTS
            print("  - loading resource:                      (LAYOUTS)")
            LAYOUTS = dd.read_csv(cp_config["Paths"]["LayoutsPath"], sep="\t")
            cols = LAYOUTS.columns
            rename = {}
            if "Container_ID_1536" in cols:
                rename["Container_ID_1536"] = "Container_Id"
            if "Conc" in cols:
                rename["Conc"] = "Conc_uM"
            if len(rename) > 0:
                LAYOUTS = LAYOUTS.rename(columns=rename)
            join_col = cp_config["Paths"]["LayoutsJoinCol"]
            name_col = cp_config["Paths"]["LayoutsNameCol"]
            LAYOUTS[join_col] = LAYOUTS[name_col] + LAYOUTS[join_col]
            drop = [name_col, "Plate_name_1536", "Address_1536", "Index", 1, 2]
            LAYOUTS = drop_cols(LAYOUTS, drop)

    else:
        raise FileNotFoundError("# unknown resource: {}".format(resource))


def drop_cols(df, cols, info=""):
    """Drops the list of columns from the DataFrame.
    Listed columns that are not present in the DataFrame are simply ignored
    (no error is thrown)."""
    df_keys = set(df.columns)
    drop = list(set(cols).intersection(df_keys))
    if len(drop) > 0:
        result = df.drop(drop, axis=1)
        print_log(result, "drop cols", info)
        return result
    else:
        return df


def well_type_from_position(df):
    """Assign the WellType from the position on the plate.
    Controls are in column 11 and 12.
    Operates inplace!"""
    df["WellType"] = "Compound"
    # df[(df["plateColumn"] == 11) | (
    #     df["plateColumn"] == 12)]["WellType"] = "Control"
    df.loc[(df["plateColumn"] == 11) | (df["plateColumn"] == 12),
           "WellType"] = "Control"
    print_log(df, "well type from pos")


def well_from_position(df, well_name="Metadata_Well",
                       row_name="plateRow", col_name="plateColumn"):
    """Assign Metadata_Well from plateRow, plateColumn.
    Operates inplace!"""
    def _well_from_position_series(s):
        return cpt.well_from_position_single(s[0], s[1])

    df[well_name] = df[[row_name, col_name]].apply(
        _well_from_position_series, axis=1)
    print_log(df, "well from pos")


def position_from_well(df, well_name="Metadata_Well",
                       row_name="plateRow", col_name="plateColumn"):
    """Generate plateRow and plateColumn from Metatadata_Well
    Operates inplace!"""
    def _position_from_well_series(well):
        return (pd.Series(cpt.position_from_well_single(well)))

    df[[row_name, col_name]] = df[well_name].apply(_position_from_well_series)
    print_log(df, "pos from well")


def get_batch_from_container(df):
    """Operates inplace!"""
    df["Batch_Id"] = df["Container_Id"].str[:9]


def get_cpd_from_container(df):
    result = pd.concat(
        [df, df["Container_Id"].str.split(":", expand=True)], axis=1)
    result = result.rename(columns={0: "Compound_Id"})
    result = drop_cols(result, [1, 2, 3, 4])
    return result


def join_layout_1536(df, plate, quadrant=""):
    """Cell Painting is always run in 384er plates.
    COMAS standard screening plates are format 1536.
    With this function, the 1536-to-384 reformatting file
    can be used directly to join the layout to the individual 384er plates.
    df is a PANDAS DataFrame."""
    assert is_pandas(df), "df has to be a Pandas DataFrame."
    join_col = cp_config["Paths"]["LayoutsJoinCol"]
    load_resource("LAYOUTS")
    # layout = LAYOUTS.copy()
    if not isinstance(quadrant, str):
        quadrant = str(quadrant)
    if len(quadrant) > 0:
        quadrant = "-" + quadrant
    result = df.copy()
    result[join_col] = plate + result["Metadata_Well"]
    result = LAYOUTS.merge(result, on=join_col, how="inner").compute()
    result = join_container(result)
    result.drop(join_col, axis=1, inplace=True)
    result["Well_Id"] = result["Container_Id"] + "_" + result["Metadata_Well"]
    result = result.apply(pd.to_numeric, errors='ignore')
    print_log(result, "join layout 1536")
    return result


# def save_ds_tmp(df, fn):
#     df.to_csv(fn, sep="\t", index=False, chunksize=CHUNKSIZE)
#     result = dd.read_csv(fn, sep="\t")
#     return result


def write_datastore(ds):
    tmp_dir = op.join(cp_config["Dirs"]["DataDir"], "tmp")
    assert len(tmp_dir) > 0, "tmp_dir may not be empty."
    cpt.create_dirs(tmp_dir)
    tmp_file = op.join(tmp_dir, "ds_tmp-*.tsv")

    ds_cols = cp_config["Paths"]["DatastoreCols"].copy()
    ds_cols.extend(ACT_PROF_PARAMETERS)
    df = ds[ds_cols]
    # df = df.sort_values("Well_Id")
    if is_pandas(df):
        df = dd.from_pandas(df, npartitions=NPARTITIONS)
    df = df.repartition(npartitions=NPARTITIONS, force=True)
    print("    - number of partitions:", df.npartitions)
    df.to_csv(tmp_file, index=False, sep="\t")
    df = dd.read_csv(tmp_file, sep="\t")
    df.to_csv(cp_config["Paths"]["DatastorePath"], index=False, sep="\t")
    print_log(df, "write datastore")


def update_datastore(df, on="Well_Id"):
    """df is a Pandas DataFrame"""
    assert is_pandas(df), "df has to be a Pandas DataFrame."
    ds = read_resource("DATASTORE")
    df2 = df.copy()
    ds_cols = cp_config["Paths"]["DatastoreCols"].copy()
    ds_cols.extend(ACT_PROF_PARAMETERS)
    df2 = df2[ds_cols]
    if is_pandas(ds):
        ds = pd.concat([ds, df2])
    else:
        ds = dd.concat([ds, df2], interleave_partitions=True)
    ds = ds.drop_duplicates(subset=on, keep="last")
    write_datastore(ds)
    print_log(df2, "update datastore")


def invert_how(how):
    """because of the Dask DF usage, `how` has to be inverted in some cases,
    where formally, not the other DF is joined onto the current df,
    but the current df is joined onto the other DF (which is a Dask DF):
    left becomes right and vice versa."""
    how = how.lower()
    if how == "left":
        how = "right"
    elif how == "right":
        how = "left"
    return how


def join_batch_data(df, df_data=None, how="left", fillna="n.d."):
    """Join data from Batch_Id.
    df is a Pandas DataFrame"""
    assert is_pandas(df), "df has to be a Pandas DataFrame."
    # bc. of the Dask DF usage, `how` has to be inverted:
    # left becomes right and vice versa
    # (formally, the df is joined onto the batch data (which is a Dask DF))
    how = invert_how(how)
    if df_data is None:
        load_resource("BATCH_DATA")
        df_data = BATCH_DATA
    if "Batch_Id" not in df.keys():
        get_batch_from_container(df)
    result = df_data.merge(df, on="Batch_Id", how=how)
    if is_dask(result):
        result = result.compute()
    result = result.apply(pd.to_numeric, errors='ignore')
    result = result.fillna(fillna)
    print_log(result, "join batch data")
    return result


def join_container_data(df, df_data=None, how="left", fillna=""):
    """Join data from Container_Id.
    df is a Pandas DataFrame"""
    assert is_pandas(df), "df has to be a Pandas DataFrame."
    how = invert_how(how)
    if df_data is None:
        load_resource("CONTAINER_DATA")
        df_data = CONTAINER_DATA
    result = df_data.merge(df, on="Container_Id", how=how)
    if is_dask(result):
        result = result.compute()
    result = result.apply(pd.to_numeric, errors='ignore')
    result = result.fillna(fillna)
    print_log(result, "join cntnr data")
    return result


def join_container(df, cont_data=None):
    """df is a Pandas DataFrame"""
    assert is_pandas(df), "df has to be a Pandas DataFrame."
    if cont_data is None:
        load_resource("CONTAINER")
        cont_data = CONTAINER
    result = cont_data.merge(df, on="Container_Id", how="inner")
    if is_dask(result):
        result = result.compute()
    print_log(result, "join container")
    return result


def join_smiles(df, df_smiles=None, how="left"):
    """Join Smiles from Compound_Id.
    df is a Pandas DataFrame, result will also be a Pandas DF."""
    assert is_pandas(df), "df has to be a Pandas DataFrame."
    how = invert_how(how)
    if df_smiles is None:
        load_resource("SMILES")
        df_smiles = SMILES
    result = df_smiles.merge(df, on="Compound_Id", how=how)
    if is_dask(result):
        result = result.compute()
    result["Compound_Id"] = result["Compound_Id"].apply(pd.to_numeric, errors='ignore')
    result["Smiles"] = result["Smiles"].fillna("*")
    result["Is_Ref"] = result["Is_Ref"].fillna(False)
    print_log(result, "join smiles")
    return result


def join_annotations(df):
    """Join Annotations from Compound_Id.
    df is a Pandas DataFrame, result will also be a Pandas DF."""
    assert is_pandas(df), "df has to be a Pandas DataFrame."
    load_resource("ANNOTATIONS")
    df = drop_cols(df, ["Is_Ref", "Trivial_Name", "Known_Act"])
    result = ANNOTATIONS.merge(df, on="Compound_Id", how="right")
    result = result.compute()  # ANNOTATIONS is Dask DF
    result = result.fillna("")
    print_log(result, "join annotations")
    return result


def extract_references(df=None):
    """Extract the references from the given df (Dask or Pandas,
    or from the DATASTORE, if df is None) by the Is_Ref tag.
    Joins the known_activities from file and saves as REFERENCES resource."""
    if df is None:
        load_resource("DATASTORE")
        df = DATASTORE
    df_ref = df[(df["Is_Ref"]) & (~df["Toxic"]) & (df["Pure_Flag"] != "Fail") &
                (df["Activity"] > 5.0)]
    if is_dask(df_ref):
        df_ref = df_ref.compute()
    df_anno = join_annotations(df_ref)
    if is_dask(df_anno):
        df_anno = df_anno.compute()
    df_anno.to_csv(cp_config["Paths"]["ReferencesPath"], sep="\t", index=False)
    print_log(df_anno, "write annotations")


def metadata(df):
    """Returns a list of the those parameters in the DataFrame that are NOT CellProfiler measurements."""
    parameters = [k for k in df.keys()
                  if not (k.startswith("Count_") or k.startswith("Median_"))]
    return parameters


def measurements(df):
    """Returns a list of the CellProfiler parameters that are in the DataFrame."""
    parameters = [k for k in df.select_dtypes(include=[np.number]).keys()
                  if k.startswith("Median_")]
    return parameters


def numeric_parameters(df):
    result = df.copy()[measurements(df)]
    return result


def flag_toxic(df, cutoff=LIMIT_CELL_COUNT_L / 100):
    """Flag data rows of toxic compounds.
    Operates inplace!"""
    median_cell_count_controls = df[df["WellType"] == "Control"]["Count_Cells"].median()
    df["Toxic"] = (df["Count_Cells"] < median_cell_count_controls * cutoff)
    df["Rel_Cell_Count"] = (
        100 * (df["Count_Cells"] / median_cell_count_controls)).astype(int)
    flagged = df["Toxic"].sum()
    print_log(df, "flag toxic", "{:3d} flagged".format(flagged))


def remove_toxic(df, cutoff=LIMIT_CELL_COUNT_L / 100):
    """Remove data rows of toxic compounds"""
    if "Toxic" not in df.keys():
        flag_toxic(df, cutoff=cutoff)
    result = df[~df["Toxic"]]
    toxic = df[df["Toxic"]]
    if is_pandas(toxic):
        info = "{:3d} removed".format(toxic.shape[0])
    else:
        info = ""
    print_log(result, "remove toxic", info)
    return result, toxic


def remove_skipped_echo_direct_transfer(df, fn):
    """Remove wells that were reported as skipped in the Echo protocol (xml).
    This functions works with Echo direct transfer protocols.
    Function supports using wildcards in the filename, the first file will be used.
    Returns a new dataframe without the skipped wells."""
    assert fn.endswith(".xml"), "Echo file expected in XML format."
    skipped_wells = []
    try:
        echo_fn = glob.glob(fn)[0]  # use the first glob match
    except IndexError:
        raise FileNotFoundError("Echo file could not be found")
    echo_print = ET.parse(echo_fn).getroot()
    skipped = echo_print.find("skippedwells")
    for well in skipped.findall("w"):
        skipped_wells.append(cpt.format_well(well.get("dn")))
    # print("Skipped wells (will be removed):", skipped_wells)
    # remove the rows with the skipped wells
    #   i.e. keep the rows where Metadata_Well is not in the list skipped_wells
    result = df[~df["Metadata_Well"].isin(skipped_wells)]
    skipped_str = "(" + ", ".join(skipped_wells) + ")"
    print_log(result, "remove skipped", "{:3d} skipped {}"
              .format(df.shape[0] - result.shape[0], skipped_str))
    return result, skipped_wells


def remove_impure(df, strict=False, reset_index=True):
    """Remove entries with `Pure_Flag == "Fail"`
    If `strict == True` compound with `Pure_Flag == Warn` are also removed."""
    pd_or_dd = pd if isinstance(df, pd.DataFrame) else dd
    outliers_list = []
    outl = df[df["Pure_Flag"] == "Fail"]
    result = df[df["Pure_Flag"] != "Fail"]
    outliers_list.append(outl)
    if strict:
        outl = result[result["Pure_Flag"] == "Warn"]
        result = result[result["Pure_Flag"] != "Warn"]
        outliers_list.append(outl)
    outliers = pd_or_dd.concat(outliers_list)
    if reset_index:
        result = result.reset_index()
        outliers = outliers.reset_index()
        result = result.drop("index", axis=1)
        outliers = outliers.drop("index", axis=1)
    if is_pandas(outliers):
        info = "{:3d} removed".format(outliers.shape[0])
    else:
        info = ""
    print_log(result, "remove impure", info)
    return result, outliers


def group_on_well(df, group_by=FINAL_PARAMETERS):
    """Group results on well level."""
    group_by = list(set(group_by).intersection(set(df.columns)))
    result = df.groupby(by=group_by).median().reset_index()
    print_log(result, "group on well")
    return result


def activity_profile(df, parameters=ACT_PROF_PARAMETERS, act_cutoff=1.58, only_final=True):
    """Calculates the log2-fold values for the parameters.
    If a list of parameters is given, then the activity profile will be
    calculated for these parameters.
    If `only_final` == `True`, then only the parameters listed in `FINAL_PARAMETERS`
    are kept in the output_table.
    df is a PANDAS DataFrame.

    Returns a new Pandas DF."""
    assert is_pandas(df), "df has to be a Pandas DataFrame."

    def _log2_mad(x, median, mad):
        if mad < 1E-4:
            mad = 1E-4
        if abs(x - median) <= (1.5 * mad):
            return 0.0
        if x >= median:
            l2f = math.log2(((x - median) / mad))
        else:
            l2f = -math.log2(((median - x) / mad))
        return l2f

    decimals = {"Activity": 1}
    result = df.copy()

    if parameters is None:  # choose all numeric parameters
        act_parameters = measurements(df)
    else:
        act_parameters = parameters.copy()
    assert len(act_parameters) > 0
    # sort parameters alphabetically
    act_parameters.sort()
    controls = df[act_parameters][df["WellType"] == "Control"]

    for key in act_parameters:
        median = controls[key].median()
        mad = controls[key].mad()
        try:
            result[key] = result[key].apply(
                lambda x: _log2_mad(x, median, mad))
        except ValueError:
            print(result[key].min(), median)
            raise

    result["Activity"] = 100 * ((result[act_parameters].abs() > act_cutoff).sum(axis=1) /
                                len(act_parameters))

    if only_final:
        r_keys = set(result.keys())
        keep = FINAL_PARAMETERS.copy()
        keep.extend(act_parameters)
        keep_and_present = list(r_keys.intersection(set(keep)))
        result = result[keep_and_present]
    result = result.round(decimals)
    print_log(result, "activity profile")
    return result


def relevant_parameters(df, ctrls_std_rel_min=0.001,
                        ctrls_std_rel_max=0.1, group_by="Plate"):
    """...std_rel...: mad relative to the median value
    df is a PANDAS DataFrame."""
    assert is_pandas(df), "df has to be a Pandas DataFrame."
    relevant_table = FINAL_PARAMETERS.copy()
    ctrl_set = set(df.keys())
    plates = sorted(set(df[group_by]))
    for plate in plates:
        debug_print("Processing plate", plate)
        controls = df[(df[group_by] == plate) & (
            df["WellType"] == "Control")].select_dtypes(include=[np.number])
        median = controls.median()
        std = controls.quantile(q=QUANT).std()

        ds = std / median >= ctrls_std_rel_min
        tmp_set = set([p for p in ds.keys() if ds[p]])
        ctrl_set.intersection_update(tmp_set)
        debug_print("ctrl_set", len(ctrl_set))

        ds = std / median <= ctrls_std_rel_max
        tmp_set = set([p for p in ds.keys() if ds[p]])
        ctrl_set.intersection_update(tmp_set)
        # debug_print("tmp_set", len(tmp_set))
        debug_print("ctrl_set", len(ctrl_set))

    relevant_table.extend(list(ctrl_set))
    debug_print("relevant_table", len(relevant_table))

    result_keys = list(df.keys())
    keep = []
    for key in result_keys:
        if key in relevant_table:
            keep.append(key)
    result = df[keep]
    debug_print("keep", len(keep))
    num_parm = len(measurements(result))
    print_log(result, "relevant parameters", "{:.3f}/{:.3f}/{:4d}"
                      .format(ctrls_std_rel_min, ctrls_std_rel_max, num_parm))
    return result


def id_filter(df, cpd_ids, id_col="Compound_Id", reset_index=True, sort_by_input=False):
    if not isinstance(cpd_ids, list):
        cpd_ids = [cpd_ids]
    result = df[df[id_col].isin(cpd_ids)]

    if reset_index:
        result = result.reset_index()
        result = result.drop("index", axis=1)
    if sort_by_input and is_pandas(df):
        result["_sort"] = pd.Categorical(
            result[id_col], categories=cpd_ids, ordered=True)
        result = result.sort_values("_sort")
        result = result.drop("_sort", axis=1)
    print_log(result, "id filter")
    return result


def find_similar(df, act_profile, cutoff=0.5, max_num=5, only_final=True,
                 parameters=ACT_PROF_PARAMETERS):
    """Filter the dataframe for activity profiles similar to the given one.
    `cutoff` gives the similarity threshold, default is 0.5.
    df can be either a Pandas OR a DASK DF.
    Returns a Pandas DF."""
    if parameters is None:  # choose all numeric parameters
        act_parameters = measurements(df)
    else:
        act_parameters = parameters.copy()
    assert len(act_parameters) > 0
    decimals = {"Similarity": 3}
    # result = df.copy()
    # df["Similarity"] = (df[act_parameters]
    #                     .apply(lambda row: [[x for x in row]], axis=1)
    #                     .apply(lambda x: cpt.profile_sim(x[0], act_profile), axis=1))  # Pandas black belt!!
    if not isinstance(act_profile, np.ndarray):
        act_profile = np.array(act_profile)
    sim_lst = []
    for _, rec in df.iterrows():
        rec_profile = rec[act_parameters].values.astype("float64")
        sim = cpt.profile_sim(act_profile, rec_profile)
        if sim >= cutoff:
            rec["Similarity"] = sim
            sim_lst.append(rec)
    if len(sim_lst) == 0:
        return pd.DataFrame()
    result = pd.DataFrame(sim_lst)
    result = result.sort_values("Similarity", ascending=False).head(max_num)
    if only_final:
        result.drop(act_parameters, axis=1, inplace=True)
    result = result.round(decimals)
    return result


def write_obj(obj, fn):
    """Save a generic python object through pickling."""
    with open(fn, "wb") as f:
        pickle.dump(obj, f)


def load_obj(fn):
    with open(fn, "rb") as f:
        obj = pickle.load(f)
    return obj


def mol_from_smiles(smi):
    if not isinstance(smi, str):
        smi = "*"
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        mol = Chem.MolFromSmiles("*")
    return mol


def write_sim_refs(sim_refs, mode="cpd"):
    """Export of sim_refs as pkl and as tsv for PPilot"""
    keep = ["Compound_Id", "Well_Id", "Is_Ref", "Ref_Id", "RefCpd_Id",
            "Similarity", "Tanimoto", "Times_Found"]
    if "ext" in mode.lower():
        sim_fn = cp_config["Paths"]["SimRefsExtPath"]
    else:
        sim_fn = cp_config["Paths"]["SimRefsPath"]
    sim_fn_pp = op.splitext(sim_fn)[0] + "_pp.tsv"
    sim_fn_pp = sim_fn_pp.replace("-*", "")  # remove Dask naming scheme
    sim_refs = sim_refs[keep]
    sim_refs.to_csv(sim_fn, sep="\t", index=False)
    # df = sim_refs.sort_values("Similarity", ascending=False)
    # df = df.drop_duplicates(subset="Well_Id", keep="first")

    # write out the highest similarity reference for PPilot:
    rec_high = {}
    for _, rec in sim_refs.iterrows():
        well_id = rec["Well_Id"]
        if well_id in rec_high:
            if rec["Similarity"] > rec_high[well_id]["Similarity"]:
                rec_high[well_id] = rec
        else:
            rec_high[well_id] = rec
    df = pd.DataFrame(list(rec_high.values()))
    df = df.rename(columns={"Similarity": "Highest_Sim"})
    df.to_csv(sim_fn_pp, sep="\t", index=False)  # tsv for PPilot
    print("* {:22s} ({:5d} |  --  )".format("write sim_refs", len(sim_refs)))


def save_sim_tmp(df_list, fn, npart=NPARTITIONS):
    if is_pandas(df_list[0]):  # empty Pandas DF
        result = pd.concat(df_list[1:])
    else:
        result = dd.concat(df_list, interleave_partitions=True)
    result = result.drop_duplicates(subset=["Well_Id", "Ref_Id"], keep="last")
    if is_pandas(result):
        result = dd.from_pandas(result, npartitions=npart)
    else:
        result = result.repartition(npartitions=npart, force=True)
    print("    - number of partitions:", result.npartitions)
    result.to_csv(fn, index=False, sep="\t")
    result = dd.read_csv(fn, sep="\t", dtype={"Smiles": np.object})
    result.to_csv(cp_config["Paths"]["SimRefsPath"], index=False, sep="\t")
    return result


def update_similar_refs(df=None):
    """Find similar compounds in references and update the export file.
    The export file of the DataFrame object is in tsv format. In addition,
    another tsv file (or maybe JSON?) is written for use in PPilot.
    `mode` can be "cpd" or "ref". if `sim_refs`is not None,
    it has to be a dict of the correct format.
    With `write=False`, the writing of the file can be deferred to the end of the processing pipeline,
    but has to be done manually, then, with `write_sim_refs()`."""
    def _chem_sim(mol_fp, query_smi):
        query = mol_from_smiles(query_smi)
        if len(query.GetAtoms()) > 1:
            query_fp = Chem.GetMorganFingerprint(query, 2)  # ECFC4
            return round(DataStructs.TanimotoSimilarity(mol_fp, query_fp), 3)
        return np.nan

    if df is None:
        load_resource("DATASTORE")
        df = DATASTORE
    load_resource("REFERENCES")
    sim_refs = read_resource("SIM_REFS")
    df_refs = REFERENCES
    tmp_dir = op.join(cp_config["Dirs"]["DataDir"], "tmp")
    assert len(tmp_dir) > 0, "tmp_dir may not be empty."
    cpt.create_dirs(tmp_dir)
    cpt.empty_dir(tmp_dir)
    tmp_file = op.join(tmp_dir, "sim_tmp-*.tsv")
    sim_refs = drop_cols(sim_refs, "Times_Found")
    ctr = cpt.Summary()
    rec_ctr = 0
    update_lst = []
    save_needed = False
    for _, rec in df.iterrows():
        if rec["Activity"] < LIMIT_ACTIVITY_L or rec["Toxic"]:
            # no similarites for low active or toxic compounds
            continue
        rec_ctr += 1
        act_profile = rec[ACT_PROF_PARAMETERS].values.astype("float64")
        max_num = 5
        if rec["Is_Ref"]:
            max_num += 1
        similar = find_similar(
            df_refs, act_profile, cutoff=LIMIT_SIMILARITY_L / 100, max_num=max_num)
        if len(similar) > 0:
            save_needed = True
            if rec["Is_Ref"]:
                similar.drop(similar.head(1).index, inplace=True)
            similar = similar[["Well_Id",
                               "Compound_Id", "Similarity", "Smiles"]]
            similar = similar.rename(
                columns={"Well_Id": "Ref_Id", "Compound_Id": "RefCpd_Id"})
            similar["Well_Id"] = rec["Well_Id"]
            similar["Is_Ref"] = rec["Is_Ref"]
            similar["Compound_Id"] = rec["Compound_Id"]
            mol = mol_from_smiles(rec.get("Smiles", "*"))
            if len(mol.GetAtoms()) > 1:
                mol_fp = Chem.GetMorganFingerprint(mol, 2)  # ECFC4
                similar["Tanimoto"] = similar["Smiles"].apply(
                    lambda q: _chem_sim(mol_fp, q))
            else:
                similar["Tanimoto"] = np.nan
            update_lst.append(similar)
            if rec_ctr % 250 == 0:
                npart = rec_ctr // 5000 + 1
                sim_refs = save_sim_tmp([sim_refs] + update_lst, tmp_file, npart=npart)
                update_lst = []
                save_needed = False

        ctr[rec["Plate"]] += 1
        if rec_ctr % 50 == 0:
            ctr["Total"] = rec_ctr
            ctr.print()
    ctr.print(final=True)

    if save_needed:
        sim_refs = save_sim_tmp([sim_refs] + update_lst, tmp_file)

    # Assign the number of times a reference was found by a research compound
    # SIM_REFS = drop_cols(SIM_REFS, ["Times_Found"])
    tmp = dd.read_csv(tmp_file, sep="\t")
    tmp = tmp[~tmp["Is_Ref"]]
    tmp = tmp.groupby(by="Ref_Id").count().reset_index()
    # "Compound_Id" is just one field that contains the correct count:
    tmp = tmp[["Ref_Id", "Compound_Id"]]
    tmp = tmp.rename(columns={"Compound_Id": "Times_Found"})
    if is_dask(tmp):
        tmp = tmp.compute()
    sim_refs = sim_refs.merge(tmp, on="Ref_Id", how="left")
    sim_refs = sim_refs.fillna(0)
    sim_refs["Times_Found"] = sim_refs["Times_Found"].astype(int)
    write_sim_refs(sim_refs)
    print_log(df, "update similar")


def well_id_similarity(df1, well_id1, df2, well_id2):
    """Calculate the similarity of the activity profiles from two compounds
    (identified by `Well_Id`). Returns value between 0 .. 1"""
    act1 = df1[df1["Well_Id"] == well_id1][ACT_PROF_PARAMETERS].values[0]
    act2 = df2[df2["Well_Id"] == well_id2][ACT_PROF_PARAMETERS].values[0]
    return round(cpt.profile_sim(act1, act2), 3)
