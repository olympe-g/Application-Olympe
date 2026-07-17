"""
Reputation Radar — early-warning monitor for digital noise around a brand.

Pipeline per mention:
  1. Sentiment (VADER)  ................ how negative is the tone?
  2. Risk-keyword scan  ................ does it touch a crisis theme (recall,
                                         contamination, lawsuit, boycott...)?
  3. Reach amplification ............... log-scaled engagement
  -> composite mention risk score (0-100)

Then, per day:
  4. Daily Risk Index .................. blend of avg negative severity and the
                                         share of high-risk mentions
  5. Volume-spike detection ............ risk-mention volume vs trailing
                                         7-day baseline (z-score-style ratio)
  -> alert level: OK / WATCH / ALERT, plus theme clustering so you can see
     WHAT is driving the spike, not just that one exists.

Run locally:   streamlit run app.py
"""

import math
import re
from datetime import timedelta

import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------- config ---

RISK_KEYWORDS = {
    # theme: (weight 1-3, patterns)
    "Product safety / contamination": (3, [
        "contaminat", "metallic", "taste[sd]? (off|like metal|weird)",
        "tastes off", "smelled odd", "corroded", "food safety", "recall",
        "batch l?\\d+", "poison", "sick", "unsafe",
    ]),
    "Legal / regulatory": (3, [
        "lawsuit", "lawyer", "legal action", "claimant", "regulator",
        "investigation", "fine[sd]?\\b", "sue[sd]?\\b",
    ]),
    "Boycott / activism": (2, [
        "boycott", "switching brands", "never buying", "hashtag trends",
        "consumer group",
    ]),
    "Commercial fallout": (2, [
        "pauses? .*promotion", "delist", "pulled from shelves",
        "retail buyer", "quality review",
    ]),
    "Trust / silence": (2, [
        "still silent", "no statement", "yet to issue", "not a good look",
        "generic reply", "furious",
    ]),
    "Service / pricing niggles": (1, [
        "refund", "late", "price .*up", "steep", "complaint",
    ]),
}

WEIGHTS = dict(sentiment=45, keywords=35, reach=20)  # sums to 100
ALERT_RED, ALERT_AMBER = 60, 35


# ------------------------------------------------------------- scoring ----

@st.cache_data
def load_and_score(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    analyzer = SentimentIntensityAnalyzer()

    def score_row(text: str, engagement: int):
        sent = analyzer.polarity_scores(text)["compound"]          # -1 .. 1
        neg = max(0.0, -sent)                                      # 0 .. 1

        kw_weight, themes = 0, []
        for theme, (w, patterns) in RISK_KEYWORDS.items():
            if any(re.search(p, text, re.IGNORECASE) for p in patterns):
                kw_weight = max(kw_weight, w)
                themes.append(theme)

        reach = min(1.0, math.log10(1 + engagement) / 3.5)         # ~2200 -> 1.0

        score = (neg * WEIGHTS["sentiment"]
                 + (kw_weight / 3) * WEIGHTS["keywords"]
                 + reach * WEIGHTS["reach"])
        return round(score, 1), sent, "; ".join(themes) or "—"

    scored = df.apply(lambda r: score_row(r["text"], r["engagement"]),
                      axis=1, result_type="expand")
    df[["risk_score", "sentiment", "themes"]] = scored
    df["date"] = df["timestamp"].dt.date
    df["is_risky"] = df["risk_score"] >= ALERT_AMBER
    return df


def daily_index(df: pd.DataFrame) -> pd.DataFrame:
    days = df.groupby("date").agg(
        mentions=("id", "count"),
        risky=("is_risky", "sum"),
        avg_risk=("risk_score", lambda s: s.nlargest(5).mean()),  # severity of worst chatter
        max_risk=("risk_score", "max"),
        top_theme=("themes", lambda s: s[s != "—"].mode().iloc[0]
                   if (s != "—").any() else "—"),
    ).reset_index()

    # volume-spike factor: risky mentions today vs trailing 7-day mean
    baseline = days["risky"].rolling(7, min_periods=3).mean().shift(1)
    days["spike_ratio"] = (days["risky"] / baseline.clip(lower=0.5)).fillna(0)

    days["risk_index"] = (
        0.5 * days["avg_risk"]
        + 0.3 * (days["risky"] / days["mentions"] * 100)
        + 0.2 * days["spike_ratio"].clip(upper=5) * 20
    ).round(1)

    def level(x):
        return "🔴 ALERT" if x >= ALERT_RED else ("🟠 WATCH" if x >= ALERT_AMBER else "🟢 OK")
    days["status"] = days["risk_index"].apply(level)
    return days


# ------------------------------------------------------------------ UI ----

st.set_page_config(page_title="Reputation Radar", page_icon="📡", layout="wide")
st.title("📡 Reputation Radar")
st.caption("Early-warning monitor for digital noise around **NordicOat** "
           "(fictional brand, self-generated dummy data)")

df = load_and_score("mentions.csv")
days = daily_index(df)
latest = days.iloc[-1]

# --- headline metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Today's Risk Index", f"{latest['risk_index']:.0f} / 100",
          delta=f"{latest['risk_index'] - days.iloc[-2]['risk_index']:+.0f} vs yesterday",
          delta_color="inverse")
c2.metric("Status", latest["status"])
c3.metric("Risky mentions today", int(latest["risky"]),
          delta=f"×{latest['spike_ratio']:.1f} vs 7-day baseline",
          delta_color="inverse")
c4.metric("Dominant theme", latest["top_theme"].split(";")[0])

if latest["risk_index"] >= ALERT_RED:
    st.error(f"**ALERT — emerging reputation risk.** Dominant theme: "
             f"*{latest['top_theme']}*. Risk-mention volume is "
             f"{latest['spike_ratio']:.1f}× the trailing baseline and tone is "
             f"deteriorating. Recommend: verify batch/QA facts, prepare a "
             f"holding statement, brief customer service, monitor hourly.")
elif latest["risk_index"] >= ALERT_AMBER:
    st.warning("**WATCH — elevated chatter.** Investigate the driving theme "
               "before it reaches mainstream media.")
else:
    st.success("Background noise within normal range.")

# --- trend chart
st.subheader("Risk Index trend")
st.line_chart(days.set_index("date")[["risk_index"]], height=220)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Mention volume: total vs risky")
    st.bar_chart(days.set_index("date")[["mentions", "risky"]], height=240)
with col_b:
    st.subheader("Theme breakdown (last 7 days)")
    recent = df[df["date"] >= days["date"].iloc[-1] - timedelta(days=6)]
    theme_counts = (recent[recent["themes"] != "—"]["themes"]
                    .str.split("; ").explode().value_counts())
    st.bar_chart(theme_counts, height=240, horizontal=True)

# --- drill-down
st.subheader("Highest-risk mentions")
n = st.slider("Show top N", 5, 30, 10)
top = (df.sort_values("risk_score", ascending=False).head(n)
       [["timestamp", "source", "text", "engagement", "risk_score", "themes"]])
st.dataframe(top, use_container_width=True, hide_index=True)

st.subheader("Daily log")
st.dataframe(days, use_container_width=True, hide_index=True)

with st.expander("How scoring works"):
    st.markdown(f"""
Each mention gets a composite **risk score (0–100)**:

| Component | Weight | Signal |
|---|---|---|
| Negative sentiment (VADER) | {WEIGHTS['sentiment']} | tone of the mention |
| Risk-keyword themes | {WEIGHTS['keywords']} | crisis vocabulary (contamination, lawsuit, boycott, recall…) weighted by severity |
| Reach | {WEIGHTS['reach']} | log-scaled engagement |

The **Daily Risk Index** blends average severity, the *share* of risky
mentions, and a **volume-spike ratio** vs the trailing 7-day baseline — so a
small but fast-growing cluster (the classic early-crisis pattern) triggers a
WATCH/ALERT even while absolute volumes are still low. Thresholds:
🟠 ≥ {ALERT_AMBER} · 🔴 ≥ {ALERT_RED}.
""")
