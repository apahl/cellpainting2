#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##
QC
##

*Created on Mon, Mar 26 by A. Pahl*

Tools for interactive plate quality control in the notebook.
Uses Holoviews and its Bokeh backend for visualization.
"""

import os.path as op
from collections import Counter

import pandas as pd
import numpy as np

from rdkit.Chem import AllChem as Chem

from IPython.core.display import HTML

import holoviews as hv
hv.extension("bokeh")
from bokeh.models import HoverTool

from cellpainting2 import processing as cpp
from cellpainting2 import reporting as cpr
from cellpainting2 import tools as cpt
cp_config = cpt.load_config("config")

ACT_PROF_PARAMETERS = cp_config["Parameters"]
ACT_CUTOFF_PERC = cp_config["Cutoffs"]["ActCutoffPerc"]
ACT_CUTOFF_PERC_H = cp_config["Cutoffs"]["ActCutoffPercH"]
ACT_CUTOFF_PERC_REF = cp_config["Cutoffs"]["ActCutoffPercRef"]
ACT_CUTOFF_PERC_REF_H = cp_config["Cutoffs"]["ActCutoffPercRefH"]
OVERACT_H = cp_config["Cutoffs"]["OverActH"]
LIMIT_ACTIVITY_H = cp_config["Cutoffs"]["LimitActivityH"]
LIMIT_ACTIVITY_L = cp_config["Cutoffs"]["LimitActivityL"]
LIMIT_CELL_COUNT_H = cp_config["Cutoffs"]["LimitCellCountH"]
LIMIT_CELL_COUNT_L = cp_config["Cutoffs"]["LimitCellCountL"]
LIMIT_SIMILARITY_H = cp_config["Cutoffs"]["LimitSimilarityH"]
LIMIT_SIMILARITY_L = cp_config["Cutoffs"]["LimitSimilarityL"]


def add_images(df):
    """Adds an Image column to the MolFrame, used for structure tooltips in plotting.
    Only works on Pandas DataFrames, does not work for Dask DataFrames
    (call `.compute()` first)."""

    assert cpp.is_pandas(df), "Only works when the data object is a Pandas DataFrame. Consider running `.compute()` first."

    def _img_method(x):
        return "data:image/png;base64,{}".format(cpr.b64_mol(cpt.mol_from_smiles(x)))

    result = df.copy()
    result["Image"] = result["Smiles"].apply(_img_method)
    return result


def process_plate_for_qc(plate_full_name, structures=True):
    plate = cpt.split_plate_name(plate_full_name)
    src_templ = op.join(cp_config["Dirs"]["PlatesDir"], "{}-{}")
    src_dir = src_templ.format(plate.date, plate.name)
    #                                                   process as Pandas
    ds_plate = cpp.read_csv(op.join(src_dir, "Results.tsv")).compute()
    ds_plate = ds_plate.group_on_well()
    ds_plate.position_from_well()  # inplace
    ds_plate = ds_plate.remove_skipped_echo_direct_transfer(op.join(src_dir, "*_print.xml"))
    ds_plate.well_type_from_position()

    ds_plate.flag_toxic()
    ds_plate = ds_plate.activity_profile(act_cutoff=1.585)

    ds_plate = ds_plate.keep_cols(cpp.FINAL_PARAMETERS)  # JUST FOR QC

    ds_plate = ds_plate.join_layout_1536(plate.name, keep_ctrls=True)
    ds_plate.data["Plate"] = "{}-{}".format(plate.date, plate.name)

    ds_plate = ds_plate.join_smiles()
    ds_plate = ds_plate.join_batch_data()
    if structures:
        data = add_images(ds_plate.data)
    else:
        data = ds_plate.data
        data["Image"] = "no struct"
    return data


def struct_hover():
    """Create a structure tooltip that can be used in Holoviews.
    Takes a MolFrame instance as parameter."""
    hover = HoverTool(
        tooltips="""
            <div>
                <div>
                    <img src="@Image" alt="Mol" width="70%"><br>
                <div>
                <div>
                    <span style="font-size: 12px; font-weight: bold;">@{}</span>
                </div>
                <div>
                    <span style="font-size: 12px;">Induction: @Activity</span>
                </div>
                <div>
                    <span style="font-size: 12px;">Rel_Cell_Count: @Rel_Cell_Count</span>
                </div>
            </div>
        """.format("Compound_Id")
    )
    return hover


def view_plate(plate, parm="Activity",
               cmap="gist_heat_r", low=0, high=50, show=True):
    if isinstance(plate, str):
        data = process_plate_for_qc(plate)
    else:
        data = plate.copy()  # already processed, Pandas DF

    hover = struct_hover()
    plot_options = {
        "width": 800, "height": 450, "legend_position": "top_left",
        "tools": [hover], "invert_yaxis": True,
        "colorbar": True,
        "colorbar_opts": {"width": 10},
    }
    plot_styles = {"size": 15, "cmap": cmap}
    vdims = ["plateRow", "Compound_Id", "Image", "Activity", "Rel_Cell_Count"]
    if parm == "Activity" or parm == "Induction":
        plot_options["color_index"] = 4
    else:
        plot_options["color_index"] = 5
    opts = {'Scatter': {'plot': plot_options, "style": plot_styles}}
    scatter_plot = hv.Scatter(data, "plateColumn", vdims=vdims, label="Plate View")
    range_args = {"plateRow": (0.5, 16.5), "plateColumn": (0.5, 24.5),
                  parm: (low, high)}
    scatter_plot = scatter_plot.redim.range(**range_args)
    # return data
    return scatter_plot(opts)
