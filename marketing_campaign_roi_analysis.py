from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"


@dataclass(frozen=True)
class CampaignConfig:
    campaign_id: str
    campaign_name: str
    channel: str
    funnel_stage: str
    daily_budget: float
    cpm: float
    ctr: float
    conversion_rate: float
    new_customer_rate: float
    avg_order_value: float
    gross_margin: float


CAMPAIGNS = [
    CampaignConfig("C001", "Google Search - High Intent", "Google Search", "Conversion", 1400, 38, 0.035, 0.058, 0.88, 96, 0.58),
    CampaignConfig("C002", "Google Shopping - Best Sellers", "Google Shopping", "Conversion", 1150, 34, 0.030, 0.048, 0.82, 88, 0.55),
    CampaignConfig("C003", "Facebook Retargeting", "Facebook", "Retargeting", 820, 22, 0.024, 0.040, 0.62, 82, 0.56),
    CampaignConfig("C004", "Facebook Prospecting - Broad", "Facebook", "Awareness", 1250, 18, 0.016, 0.020, 0.91, 76, 0.53),
    CampaignConfig("C005", "Instagram Creator Ads", "Instagram", "Consideration", 900, 20, 0.015, 0.018, 0.87, 72, 0.52),
    CampaignConfig("C006", "TikTok Video Views", "TikTok", "Awareness", 760, 14, 0.020, 0.010, 0.94, 66, 0.50),
    CampaignConfig("C007", "Email Winback", "Email", "Retention", 900, 18, 0.040, 0.060, 0.18, 74, 0.61),
    CampaignConfig("C008", "Affiliate Partner Deals", "Affiliate", "Conversion", 620, 18, 0.033, 0.055, 0.76, 84, 0.54),
    CampaignConfig("C009", "LinkedIn B2B Trial Push", "LinkedIn", "Experiment", 980, 46, 0.010, 0.015, 0.96, 112, 0.57),
    CampaignConfig("C010", "Instagram Story Sale", "Instagram", "Promotion", 640, 18, 0.020, 0.024, 0.79, 68, 0.50),
]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)


def generate_campaign_data(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    rows = []

    for config in CAMPAIGNS:
        for date in dates:
            weekday_factor = 1.08 if date.dayofweek in [1, 2, 3] else 0.94 if date.dayofweek in [5, 6] else 1.0
            month_factor = 1.12 if date.month in [3, 5] else 0.92 if date.month == 1 else 1.0
            promo_factor = 1.18 if date.day in [1, 15, 28] and config.funnel_stage in ["Promotion", "Retargeting"] else 1.0

            spend = config.daily_budget * rng.uniform(0.82, 1.18) * weekday_factor
            impressions = spend / config.cpm * 1000 * rng.lognormal(0, 0.06)
            clicks = impressions * config.ctr * rng.lognormal(0, 0.09)
            conversions = clicks * config.conversion_rate * month_factor * promo_factor * rng.lognormal(0, 0.10)
            new_customers = conversions * config.new_customer_rate * rng.lognormal(0, 0.05)
            revenue = conversions * config.avg_order_value * rng.lognormal(0, 0.07)
            gross_profit = revenue * config.gross_margin

            rows.append(
                {
                    "date": date,
                    "campaign_id": config.campaign_id,
                    "campaign_name": config.campaign_name,
                    "channel": config.channel,
                    "funnel_stage": config.funnel_stage,
                    "spend": round(spend, 2),
                    "impressions": int(round(impressions)),
                    "clicks": int(round(clicks)),
                    "conversions": round(conversions, 1),
                    "new_customers": round(new_customers, 1),
                    "revenue": round(revenue, 2),
                    "gross_profit": round(gross_profit, 2),
                }
            )

    campaign_data = pd.DataFrame(rows)
    campaign_data.to_csv(DATA_DIR / "marketing_campaign_data.csv", index=False)
    return campaign_data


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["ctr"] = result["clicks"] / result["impressions"]
    result["conversion_rate"] = result["conversions"] / result["clicks"]
    result["cpc"] = result["spend"] / result["clicks"]
    result["cpa"] = result["spend"] / result["conversions"]
    result["cac"] = result["spend"] / result["new_customers"]
    result["roas"] = result["revenue"] / result["spend"]
    result["roi"] = (result["gross_profit"] - result["spend"]) / result["spend"]
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def summarize_performance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped_cols = {
        "spend": "sum",
        "impressions": "sum",
        "clicks": "sum",
        "conversions": "sum",
        "new_customers": "sum",
        "revenue": "sum",
        "gross_profit": "sum",
    }

    campaign_summary = df.groupby(["campaign_id", "campaign_name", "channel", "funnel_stage"], as_index=False).agg(grouped_cols)
    channel_summary = df.groupby("channel", as_index=False).agg(grouped_cols)

    campaign_summary = add_metrics(campaign_summary)
    channel_summary = add_metrics(channel_summary)

    campaign_summary["net_profit_after_marketing"] = campaign_summary["gross_profit"] - campaign_summary["spend"]
    channel_summary["net_profit_after_marketing"] = channel_summary["gross_profit"] - channel_summary["spend"]

    campaign_summary["recommendation"] = campaign_summary.apply(classify_campaign, axis=1)
    campaign_summary["decision_reason"] = campaign_summary.apply(decision_reason, axis=1)

    channel_summary["budget_action"] = channel_summary.apply(classify_channel, axis=1)

    budget_plan = build_budget_reallocation_plan(campaign_summary)

    round_cols = ["spend", "conversions", "new_customers", "revenue", "gross_profit", "ctr", "conversion_rate", "cpc", "cpa", "cac", "roas", "roi", "net_profit_after_marketing"]
    campaign_summary[round_cols] = campaign_summary[round_cols].round(3)
    channel_summary[round_cols] = channel_summary[round_cols].round(3)
    budget_plan[["current_spend", "recommended_spend", "budget_change", "projected_revenue_change", "projected_profit_change"]] = budget_plan[
        ["current_spend", "recommended_spend", "budget_change", "projected_revenue_change", "projected_profit_change"]
    ].round(2)

    campaign_summary.to_csv(OUTPUT_DIR / "campaign_performance_summary.csv", index=False)
    channel_summary.to_csv(OUTPUT_DIR / "channel_performance_summary.csv", index=False)
    budget_plan.to_csv(OUTPUT_DIR / "budget_reallocation_plan.csv", index=False)
    return campaign_summary, channel_summary, budget_plan


def classify_campaign(row: pd.Series) -> str:
    if row["roi"] >= 0.45 and row["cac"] <= 45:
        return "Scale"
    if row["roi"] < 0 or row["cac"] >= 95:
        return "Stop"
    if row["roas"] >= 1.8 and row["conversion_rate"] >= 0.035:
        return "Optimize"
    return "Reduce / Test New Creative"


def decision_reason(row: pd.Series) -> str:
    if row["recommendation"] == "Scale":
        return "Strong ROI and efficient customer acquisition."
    if row["recommendation"] == "Stop":
        return "Negative ROI or CAC is too high to justify continued spend."
    if row["recommendation"] == "Optimize":
        return "Revenue is promising, but CAC or ROI can improve with targeting and landing-page tests."
    return "Weak efficiency; lower budget until creative, audience, or offer improves."


def classify_channel(row: pd.Series) -> str:
    if row["roi"] >= 0.35 and row["cac"] <= 55:
        return "Invest More"
    if row["roi"] < 0:
        return "Cut Budget"
    return "Hold and Optimize"


def build_budget_reallocation_plan(campaign_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in campaign_summary.iterrows():
        if row["recommendation"] == "Scale":
            multiplier = 1.25
        elif row["recommendation"] == "Optimize":
            multiplier = 1.05
        elif row["recommendation"] == "Stop":
            multiplier = 0.15
        else:
            multiplier = 0.65

        current_spend = row["spend"]
        recommended_spend = current_spend * multiplier
        budget_change = recommended_spend - current_spend
        marginal_efficiency = max(row["roas"] * 0.82, 0)
        projected_revenue_change = budget_change * marginal_efficiency
        projected_profit_change = projected_revenue_change * (row["gross_profit"] / row["revenue"]) - budget_change if row["revenue"] else -budget_change

        rows.append(
            {
                "campaign_name": row["campaign_name"],
                "channel": row["channel"],
                "recommendation": row["recommendation"],
                "current_spend": current_spend,
                "recommended_spend": recommended_spend,
                "budget_change": budget_change,
                "projected_revenue_change": projected_revenue_change,
                "projected_profit_change": projected_profit_change,
            }
        )

    return pd.DataFrame(rows)


def build_charts(campaign_summary: pd.DataFrame, channel_summary: pd.DataFrame, budget_plan: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    channel_sorted = channel_summary.sort_values("roi", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2f7d5c" if value >= 0.35 else "#d08b36" if value >= 0 else "#b94f45" for value in channel_sorted["roi"]]
    ax.barh(channel_sorted["channel"], channel_sorted["roi"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Marketing ROI by Channel", fontsize=14, fontweight="bold")
    ax.set_xlabel("ROI based on gross profit")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "roi_by_channel.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(channel_summary))
    width = 0.36
    ax.bar(x - width / 2, channel_summary["spend"], width, label="Spend", color="#6b7280")
    ax.bar(x + width / 2, channel_summary["revenue"], width, label="Revenue", color="#2f6f73")
    ax.set_xticks(x)
    ax.set_xticklabels(channel_summary["channel"], rotation=20, ha="right")
    ax.set_title("Spend vs Revenue by Channel", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "spend_vs_revenue_by_channel.png", dpi=180)
    plt.close(fig)

    campaign_sorted = campaign_summary.sort_values("cac", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#b94f45" if rec == "Stop" else "#d08b36" if "Reduce" in rec else "#2f7d5c" if rec == "Scale" else "#4f79a7" for rec in campaign_sorted["recommendation"]]
    ax.barh(campaign_sorted["campaign_name"], campaign_sorted["cac"], color=colors)
    ax.set_title("Customer Acquisition Cost by Campaign", fontsize=14, fontweight="bold")
    ax.set_xlabel("CAC")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "cac_by_campaign.png", dpi=180)
    plt.close(fig)

    plan_sorted = budget_plan.sort_values("budget_change")
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#b94f45" if value < 0 else "#2f7d5c" for value in plan_sorted["budget_change"]]
    ax.barh(plan_sorted["campaign_name"], plan_sorted["budget_change"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Recommended Budget Reallocation", fontsize=14, fontweight="bold")
    ax.set_xlabel("Budget change")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "budget_reallocation.png", dpi=180)
    plt.close(fig)


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    text_df = df.copy().astype(str)
    headers = list(text_df.columns)
    rows = text_df.values.tolist()
    widths = [
        max(len(header), *(len(row[col_idx]) for row in rows)) if rows else len(header)
        for col_idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    divider_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body_lines = [
        "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider_line, *body_lines])


def write_report(campaign_summary: pd.DataFrame, channel_summary: pd.DataFrame, budget_plan: pd.DataFrame) -> None:
    total_spend = campaign_summary["spend"].sum()
    total_revenue = campaign_summary["revenue"].sum()
    total_gross_profit = campaign_summary["gross_profit"].sum()
    total_roi = (total_gross_profit - total_spend) / total_spend
    blended_cac = total_spend / campaign_summary["new_customers"].sum()

    scale_campaigns = campaign_summary[campaign_summary["recommendation"] == "Scale"].sort_values("roi", ascending=False)
    stop_campaigns = campaign_summary[campaign_summary["recommendation"] == "Stop"].sort_values("roi")
    best_channel = channel_summary.sort_values("roi", ascending=False).iloc[0]
    worst_channel = channel_summary.sort_values("roi").iloc[0]
    projected_profit_change = budget_plan["projected_profit_change"].sum()

    compact_campaign = campaign_summary[
        [
            "campaign_name",
            "channel",
            "spend",
            "new_customers",
            "cac",
            "roas",
            "roi",
            "net_profit_after_marketing",
            "recommendation",
            "decision_reason",
        ]
    ].sort_values(["recommendation", "roi"], ascending=[True, False])

    compact_channel = channel_summary[
        [
            "channel",
            "spend",
            "revenue",
            "new_customers",
            "cac",
            "roas",
            "roi",
            "net_profit_after_marketing",
            "budget_action",
        ]
    ].sort_values("roi", ascending=False)

    lines = [
        "# Marketing Campaign Performance + ROI Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"The campaigns generated **{money(total_revenue)}** in revenue from **{money(total_spend)}** in marketing spend.",
        f"Blended CAC was **{money(blended_cac)}**, and total marketing ROI based on gross profit was **{pct(total_roi)}**.",
        f"The strongest channel is **{best_channel['channel']}** with ROI of **{pct(best_channel['roi'])}** and CAC of **{money(best_channel['cac'])}**.",
        f"The weakest channel is **{worst_channel['channel']}** with ROI of **{pct(worst_channel['roi'])}** and CAC of **{money(worst_channel['cac'])}**.",
        f"The budget reallocation scenario is estimated to change profit by **{money(projected_profit_change)}** over the analysis period.",
        "",
        "## Where To Invest More",
        "",
        dataframe_to_markdown(scale_campaigns[["campaign_name", "channel", "cac", "roas", "roi", "net_profit_after_marketing"]]),
        "",
        "## Campaigns To Stop",
        "",
        dataframe_to_markdown(stop_campaigns[["campaign_name", "channel", "cac", "roas", "roi", "net_profit_after_marketing", "decision_reason"]]),
        "",
        "## Campaign Performance",
        "",
        dataframe_to_markdown(compact_campaign),
        "",
        "## Channel Performance",
        "",
        dataframe_to_markdown(compact_channel),
        "",
        "## Budget Reallocation Plan",
        "",
        dataframe_to_markdown(budget_plan.sort_values("budget_change")),
        "",
        "## Business Recommendations",
        "",
        "1. Increase budget for campaigns with strong ROI and efficient CAC, especially high-intent search, affiliate, retargeting, and email.",
        "2. Stop or sharply reduce campaigns with negative ROI and high CAC, then only restart them with a new audience, offer, or creative hypothesis.",
        "3. Keep a small testing budget for awareness channels, but judge them separately from bottom-funnel conversion campaigns.",
        "4. Move reporting from campaign-level vanity metrics to CAC, ROAS, gross-profit ROI, and net profit after marketing.",
        "5. Review budget allocation weekly so spend follows performance instead of historical channel habits.",
        "",
        "## Resume Talking Point",
        "",
        "Built a marketing ROI analysis project using Python to evaluate CAC, ROAS, gross-profit ROI, channel performance, and campaign-level budget decisions, identifying which campaigns to scale, optimize, reduce, or stop.",
        "",
        "## Charts",
        "",
        "- `outputs/charts/roi_by_channel.png`",
        "- `outputs/charts/spend_vs_revenue_by_channel.png`",
        "- `outputs/charts/cac_by_campaign.png`",
        "- `outputs/charts/budget_reallocation.png`",
    ]

    (OUTPUT_DIR / "marketing_roi_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    data = generate_campaign_data()
    campaign_summary, channel_summary, budget_plan = summarize_performance(data)
    build_charts(campaign_summary, channel_summary, budget_plan)
    write_report(campaign_summary, channel_summary, budget_plan)
    print("Marketing campaign ROI analysis complete.")
    print(f"Report: {OUTPUT_DIR / 'marketing_roi_report.md'}")
    print(f"Charts: {CHART_DIR}")


if __name__ == "__main__":
    main()
