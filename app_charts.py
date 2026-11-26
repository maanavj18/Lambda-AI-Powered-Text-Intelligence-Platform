"""Altair charts for the About tab of the Streamlit app."""
import altair as alt
import pandas as pd

# Colors: blue and orange are a colorblind-safe pair; gray marks "left out" or "baseline".
POSITIVE = "#2a78d6"
NEGATIVE = "#eb6834"
NEUTRAL = "#a3a29c"

BAR_RADIUS = 4


def star_distribution(score_counts: dict[int, int]) -> alt.Chart:
    """Reviews per star rating, colored by the sentiment label each rating maps to."""
    def label(stars: int) -> str:
        return "Positive" if stars >= 4 else "Negative" if stars <= 2 else "Left out (3 stars)"

    total = sum(score_counts.values())
    df = pd.DataFrame({
        "Stars": [f"{s} star{'s' if s > 1 else ''}" for s in score_counts],
        "Reviews": list(score_counts.values()),
        "Label": [label(s) for s in score_counts],
    })
    df["Share"] = df.Reviews / total

    base = alt.Chart(df).encode(
        x=alt.X("Stars:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Reviews:Q", title="Reviews"),
    )
    bars = base.mark_bar(cornerRadiusTopLeft=BAR_RADIUS, cornerRadiusTopRight=BAR_RADIUS).encode(
        color=alt.Color(
            "Label:N",
            scale=alt.Scale(
                domain=["Negative", "Left out (3 stars)", "Positive"],
                range=[NEGATIVE, NEUTRAL, POSITIVE],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("Stars:N"),
            alt.Tooltip("Reviews:Q", format=","),
            alt.Tooltip("Share:Q", format=".1%"),
            alt.Tooltip("Label:N", title="Labeled as"),
        ],
    )
    # Mid gray stays readable on both the light and the dark Streamlit theme.
    labels = base.mark_text(dy=-8, color="#8a8985").encode(text=alt.Text("Reviews:Q", format=","))
    return (bars + labels).properties(height=320)


def confusion_matrix(matrix: list[list[int]], title: str) -> alt.Chart:
    """2x2 heatmap. Rows are the true label, columns the predicted label."""
    labels = ["Negative", "Positive"]
    total = sum(sum(row) for row in matrix)
    rows = [
        {
            "Actual": actual,
            "Predicted": predicted,
            "Reviews": matrix[i][j],
            "Share": matrix[i][j] / total,
            "Outcome": "Correct" if i == j else "Wrong",
        }
        for i, actual in enumerate(labels)
        for j, predicted in enumerate(labels)
    ]
    df = pd.DataFrame(rows)
    peak = max(max(row) for row in matrix)

    base = alt.Chart(df).encode(
        x=alt.X("Predicted:N", sort=labels, title="Predicted", axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("Actual:N", sort=labels, title="Actual"),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=BAR_RADIUS).encode(
        color=alt.Color(
            "Reviews:Q",
            scale=alt.Scale(scheme="blues", domain=[0, peak]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Actual:N"),
            alt.Tooltip("Predicted:N"),
            alt.Tooltip("Outcome:N"),
            alt.Tooltip("Reviews:Q", format=","),
            alt.Tooltip("Share:Q", format=".1%", title="Share of test set"),
        ],
    )
    text = base.mark_text(fontSize=16, fontWeight="bold").encode(
        text=alt.Text("Reviews:Q", format=","),
        color=alt.condition(alt.datum.Reviews > peak / 2, alt.value("white"), alt.value("#0b0b0b")),
    )
    return (cells + text).properties(title=title, height=220)
