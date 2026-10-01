import pandas as pd

from composer_clf.demo_charts import probability_chart


def test_probability_chart_top5_spec():
    probs = pd.Series([0.05, 0.4, 0.1, 0.2, 0.15, 0.06, 0.04],
                      index=["a", "b", "c", "d", "e", "f", "g"]).sort_values(ascending=False)
    spec = probability_chart(probs).to_dict()
    values = spec["datasets"][spec["data"]["name"]]
    assert [v["composer"] for v in values] == ["b", "d", "e", "c", "f"]
    bars, labels = spec["layer"]
    assert bars["mark"]["type"] == "bar" and bars["mark"]["color"] == "#2a78d6"
    assert bars["mark"]["cornerRadiusEnd"] == 4 and bars["mark"]["size"] <= 24
    assert {t["field"] for t in bars["encoding"]["tooltip"]} == {"composer", "probability"}
    assert labels["mark"]["type"] == "text" and labels["mark"]["color"] != "#2a78d6"


def test_roll_image_puts_high_pitches_on_top():
    import numpy as np

    from composer_clf.demo_charts import roll_image

    roll = np.zeros((64, 1536), np.uint8)
    roll[0, :10] = 127                       # lowest pitch, loudest velocity
    img = roll_image(roll)
    assert img.shape == (64, 1536) and img.dtype == np.uint8
    assert (img[-1, :10] == 254).all() and img[0].sum() == 0


def test_roll_image_can_stretch_pitch_rows():
    import numpy as np

    from composer_clf.demo_charts import roll_image

    roll = np.zeros((64, 1536), np.uint8)
    roll[63] = 100                            # highest pitch
    img = roll_image(roll, row_scale=4)
    assert img.shape == (256, 1536)
    assert (img[:4] == 200).all() and img[4:].sum() == 0
