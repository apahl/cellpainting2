#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##############
Resource Paths
##############

*Created on Sun Aug 6, 2017 18:30 by A. Pahl*"""


src_path = "/home/pahl/comas/projects/painting/plates/{}-{}"
smiles_path = "/home/pahl/comas/share/comas_smiles_b64.tsv.gz"    # tab-delim., gzipped
smiles_cols = ['Compound_Id', "Smiles"]
batch_path = "/home/pahl/comas/share/comas_batch.tsv.gz"    # tab-delim., gzipped
batch_cols = []
container_path = "/home/pahl/comas/share/comas_container.tsv.gz"    # tab-delim., gzipped
container_cols = []
# tab-delim., gzipped
batch_data_path = "/home/pahl/comas/share/comas_batch_data.tsv.gz"
batch_data_cols = ['Batch_Id', "Pure_Flag"]
# tab-delim., gzipped
container_data_path = "/home/pahl/comas/share/comas_container_data.tsv.gz"
container_data_cols = []
annotations_path = "/home/pahl/comas/share/known_act.tsv"  # tab-delim.
references_path = "/home/pahl/comas/notebooks/projects/painting2/data/references_act_prof.tsv"
sim_refs_path = "/home/pahl/comas/notebooks/projects/painting2/data/sim_refs.tsv"
sim_refs_ext_path = "/home/pahl/comas/notebooks/projects/painting2/data/sim_refs_ext.tsv"
datastore_path = "/home/pahl/comas/notebooks/projects/painting2/data/cp_datastore.tsv"
datastore_cols = ["Compound_Id", "Container_Id", "Well_Id", "Producer", "Conc_uM", "Is_Ref",
                  "Activity", "Toxic", "Pure_Flag", "Rel_Cell_Count", "Metadata_Well", "Plate",
                  "Smiles"]

layouts_path = "/home/pahl/comas/projects/painting/plates/layouts/layouts.tsv"

QUADRANTS = {  # measured quadrants, should be 1..4 for all but the last plate.
    "SI0009": ["1", "2", "3", "4"],
    "SI0012": ["1", "2", "3", "4"],
    "C2017": ["01", "02", "03", "04"],
}

DATES = {
    "S0195-1": "171121",
    "S0195-2": "171121",
    "S0195-3": "171122",
    "S0195-4": "171122",
    "S0198-1": "171122",
    "S0198-2": "171122",
    "S0198-3": "171122",
    "S0198-4": "171122",
    "S0203-1": "171122",
    "S0203-2": "171122",
    "S0203-3": "171122",
    "S0203-4": "171122",
    "SI0009-1": "171122",
    "SI0009-2": "171122",
    "SI0009-3": "171122",
    "SI0009-4": "171122",
    "SI0012-1": "171130",
    "SI0012-2": "171130",
    "SI0012-3": "171130",
    "SI0012-4": "171130",
    "C2017-01": "171208",
    "C2017-02": "171209",
    "C2017-03": "171209",
    "C2017-04": "171213",
}
