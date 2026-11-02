"""
Client Database — maps each client to its dataset and capabilities.

To add a client: add one entry to CLIENT_DB.
has_embeddings=False means multiplier-only (like Toasttab).
"""
import os
DIR = os.path.dirname(__file__)

CLIENT_DB = {
    "tekion":   {"name":"Tekion",   "dataset":os.path.join(DIR,"Tekion_dataset.xlsx"),            "has_embeddings":True},
    "clario":   {"name":"Clario",   "dataset":os.path.join(DIR,"Clario_Wastage_dataset.xlsx"),    "has_embeddings":True, "sheet":"Sheet1"},
    "odessia":  {"name":"Odessia",  "dataset":os.path.join(DIR,"Odessia Dataset.xlsx"),           "has_embeddings":True},
    "rippling": {"name":"Rippling", "dataset":os.path.join(DIR,"Rippling_Dataset.xlsx"),          "has_embeddings":True},
    "stripe":   {"name":"Stripe",   "dataset":os.path.join(DIR,"Stripe_Dataset.xlsx"),            "has_embeddings":True},
    "tessolve": {"name":"Tessolve", "dataset":os.path.join(DIR,"Tesslove_Wastage_Dataset.xlsx"),  "has_embeddings":True},
    "toasttab": {"name":"Toasttab", "dataset":None, "has_embeddings":False},
}



CLIENT_LIST = [v["name"] for v in CLIENT_DB.values()]

def name_to_key(display_name):
    for k, v in CLIENT_DB.items():
        if v["name"] == display_name: return k
    raise ValueError(f"Unknown client '{display_name}'")

def get_info(ck):
    return CLIENT_DB[ck.strip().lower()]
