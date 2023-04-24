from datetime import datetime
from io import StringIO
import pytz
import os

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import altair as alt

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

FEATURES = [
    "price",
    "previous_month_sales",
    "advertising_budget",
    "retailer_discount",
    "product_innovation",
    "market_demand",
    "manufacturer_reputation",
    "seasonal_demand",
    "bulk_purchase_discount",
    "brand_loyalty",
    "product_age",
    "in_store_promotion",
    "product_visibility",
    "customer_reviews",
    "stock_availability",
    "online_sales_channel",
    "industry_trend",
    "competitor_activity",
    "retailer_partnership",
    "social_media_engagement",
]

EARLIEST_DATE = datetime(2022, 3, 1).astimezone(pytz.utc)


palette = {
    "primary": {
        "white": "#ffffff",
        "black": "#000000",
        "union500": "#43474e",
        "union400": "#67696d",
        "union300": "#C1C3C6",
        "union200": "#e6e7e8",
        "union100": "#f2f3f3",
        "yellow": "#fcb51f",
        "yellowLight1": "#F9D48E",
        "yellowLight2": "#FFF6E4",
        "yellowDark": "#E79D00",
        "purple": "#541dff",
        "lightBlue": "#bfe6ff",
        "blue": "#009dff",
    },
    "state": {
        "succeeded": "#77c332",
        "succeeding": "#c0d765",
        "aborted": "#c91dff",
        "aborting": "#ce90ff",
        "failed": "#ee678e",
        "failing": "#fc8d14",
        "running": "#456fff",
        "queued": "#fcb51f",
        "undefined": "#b87428",
        "nested": "#aaaaaa",
        "timedout": "#feabab",
    },
}

def log_normalize(x):
    pos_log_x = np.log(1 + x)
    return pos_log_x / pos_log_x.sum()


def load_items_df() -> pd.DataFrame:
    items_df = pd.read_parquet(os.path.join(CURRENT_DIR, "items.parquet"))
    items_df["weight"] = log_normalize(items_df.weight)
    return items_df


def bar_plot_html(x, y, title=None, x_label=None, y_label=None):
    # Create the bar chart
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            orientation="h",
        )
    )

    # Customize the plot
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        font=dict(size=12),
        plot_bgcolor="#f0f0f0",
        xaxis=dict(gridcolor="#e0e0e0", gridwidth=1, zeroline=False),
        yaxis=dict(
            gridcolor="#e0e0e0", gridwidth=1, zeroline=False, autorange="reversed"
        ),
    )

    fig.show()

    return plotly.io.to_html(fig, full_html=False)


def bar_plot_altair_html(data, x_label, y_label):

    # Bar plot
    chart = alt.Chart(data).mark_bar(color="#D093F1").encode(
        y=alt.Y(f'{y_label}:N', sort='-x'),
        x=alt.X(f'{x_label}:Q'),
    ).properties(width=600, height=500)

    str_io = StringIO()

    chart.save(str_io, format="html", embed_options={'renderer': 'svg'})
    return str_io.getvalue()