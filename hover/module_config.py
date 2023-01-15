import hover
import bokeh.palettes

# constants for the abstain mechanism
ABSTAIN_DECODED = hover.config["data.values"]["abstain_decoded"]
ABSTAIN_ENCODED = hover.config["data.values"]["abstain_encoded"]
ABSTAIN_HEXCOLOR = hover.config["visual"]["abstain_hexcolor"]
BOKEH_PALETTE = getattr(bokeh.palettes, hover.config["visual"]["bokeh_palette_name"])

# constants for label encoding mechanism
ENCODED_LABEL_KEY = hover.config["data.columns"]["encoded_label_key"]

# constants for saving work
DATA_SAVE_DIR = hover.config["io"]["data_save_dir"]
