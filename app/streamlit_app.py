"""Upload a MIDI file and see which composer the model predicts, and why."""
import os
import tempfile
from pathlib import Path

import joblib
import streamlit as st

from composer_clf import paths
from composer_clf.features.jsymbolic import extract_jsymbolic
from composer_clf.predict import explain

MODEL = Path(os.environ.get("COMPOSER_MODEL", paths.MODELS / "lightgbm_hf.joblib"))

st.set_page_config(page_title="Composer classifier")
st.title("Who composed this?")


@st.cache_resource
def load_bundle(path: Path):
    return joblib.load(path)


bundle = load_bundle(MODEL)
st.caption(f"LightGBM on {len(bundle['feature_columns']):,} jSymbolic features. "
           f"Composers: {', '.join(bundle['labels'])}")
upload = st.file_uploader("MIDI file", type=["mid", "midi"])
if upload:
    with tempfile.TemporaryDirectory() as tmp:
        midi = Path(tmp) / upload.name
        midi.write_bytes(upload.getvalue())
        with st.spinner("Extracting jSymbolic features (about 10 seconds)"):
            feats, failed = extract_jsymbolic([midi], Path(tmp) / "work", n_shards=1, workers=1, xmx="2g")
        if failed or feats.empty:
            st.error("jSymbolic could not read this file. Try another MIDI file.")
            st.stop()
        try:
            probs, top = explain(bundle, feats.drop(columns="path"))
        except ValueError as err:
            st.error(f"Could not score this file: {err}")
            st.stop()
        st.subheader(f"Prediction: {probs.index[0]} ({probs.iloc[0]:.0%})")
        st.markdown("**Features that pushed the prediction most**")
        st.dataframe(top, hide_index=True)
