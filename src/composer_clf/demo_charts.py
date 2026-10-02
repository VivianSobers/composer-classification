"""Charts for the demo app."""
import altair as alt
import numpy as np
import pandas as pd

from composer_clf.style import BASELINE, GRID, INK_MUTED, INK_SECONDARY, SERIES, SURFACE


def probability_chart(probs: pd.Series, top_k: int = 5) -> alt.Chart:
    """Horizontal bars of the top composer probabilities, value at each bar tip, hover tooltip."""
    data = probs.sort_values(ascending=False).head(top_k).rename_axis("composer").reset_index(name="probability")
    y = alt.Y("composer:N", sort=None, title=None,
              axis=alt.Axis(labelColor=INK_SECONDARY, ticks=False, domainColor=BASELINE, labelFontSize=13))
    x = alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title=None,
              axis=alt.Axis(format="%", labelColor=INK_MUTED, gridColor=GRID, domain=False, ticks=False,
                            tickCount=5))
    tooltip = [alt.Tooltip("composer:N"), alt.Tooltip("probability:Q", format=".1%")]
    bars = alt.Chart(data).mark_bar(color=SERIES[0], size=20, cornerRadiusEnd=4).encode(x=x, y=y, tooltip=tooltip)
    labels = alt.Chart(data).mark_text(align="left", dx=6, color=INK_SECONDARY, fontSize=12).encode(
        x=x, y=y, text=alt.Text("probability:Q", format=".0%"))
    return (alt.layer(bars, labels)
            .properties(height=36 * len(data), background=SURFACE)
            .configure_view(stroke=None))


def roll_image(roll: np.ndarray, row_scale: int = 1) -> np.ndarray:
    """Grayscale image of a piano roll: high pitches on top, velocity 0-127 stretched to 0-254.

    row_scale repeats each pitch row, since a 64 x 1536 roll is too flat to read at page width.
    """
    img = (np.flipud(roll).astype(np.uint16) * 2).astype(np.uint8)
    return np.repeat(img, row_scale, axis=0)
