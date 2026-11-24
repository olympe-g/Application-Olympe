# 📡 Reputation Radar — early-warning monitor for digital noise

A lightweight AI prototype that scans a stream of online mentions (social,
forums, reviews, news) and flags **emerging reputation risks before they hit
mainstream media**.

Built with Streamlit + VADER sentiment analysis on a **self-generated dummy
dataset** (`mentions.csv`): 219 mentions over 21 days for a fictional oat-drink
brand, *NordicOat*, including a simulated slow-burn "contamination rumour" that
starts as a few Reddit posts and escalates — the classic early-crisis pattern
the tool is designed to catch. No confidential or real-brand data is used.

## How it works

**Per mention — composite risk score (0–100):**
| Signal | Weight | Method |
|---|---|---|
| Negative sentiment | 45 | VADER compound score |
| Crisis vocabulary | 35 | Weighted keyword themes (contamination, recall, lawsuit, boycott, delisting, brand silence…) |
| Reach | 20 | Log-scaled engagement |

**Per day — Daily Risk Index & alerts:**
- Severity of the day's *worst* mentions (top-5 mean)
- Share of risky mentions
- **Volume-spike ratio** vs a trailing 7-day baseline — so a small but
  fast-growing cluster triggers 🟠 WATCH / 🔴 ALERT even while absolute
  volumes are still low
- Theme clustering shows *what* is driving the spike, and an ALERT ships with
  a recommended action (verify facts, holding statement, brief CS, monitor hourly)

In the demo data the monitor turns 🟠 WATCH on the very first day of whispers
(2 Reddit posts) and 🔴 ALERT two days later — roughly a week before the story
reaches news outlets in the simulation.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — the Streamlit monitoring app
- `mentions.csv` — self-created dummy dataset (public, shareable)
- `generate_data.py` — script that generated the dataset (reproducible)
- `requirements.txt` — dependencies for Streamlit Cloud

## Next steps (production direction)

Swap the CSV for live inputs (X/Reddit/news APIs, Google Alerts RSS,
review-site scrapes), replace keyword themes with an LLM classifier for nuance
(irony, non-English mentions), add per-stakeholder views (investors,
consumers, regulators) and Slack/email alert delivery.
