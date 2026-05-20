from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"


st.set_page_config(page_title="Marketing Campaign ROI Analysis", layout="wide")


@st.cache_data
def load_data():
    campaign_data = pd.read_csv(DATA_DIR / "marketing_campaign_data.csv", parse_dates=["date"])
    campaign_summary = pd.read_csv(OUTPUT_DIR / "campaign_performance_summary.csv")
    channel_summary = pd.read_csv(OUTPUT_DIR / "channel_performance_summary.csv")
    budget_plan = pd.read_csv(OUTPUT_DIR / "budget_reallocation_plan.csv")
    return campaign_data, campaign_summary, channel_summary, budget_plan


required_file = OUTPUT_DIR / "campaign_performance_summary.csv"
if not required_file.exists():
    st.error("Run `python src/marketing_campaign_roi_analysis.py` before opening the dashboard.")
    st.stop()

campaign_data_df, campaign_summary_df, channel_summary_df, budget_plan_df = load_data()

total_spend = campaign_summary_df["spend"].sum()
total_revenue = campaign_summary_df["revenue"].sum()
total_profit = campaign_summary_df["gross_profit"].sum() - total_spend
blended_cac = total_spend / campaign_summary_df["new_customers"].sum()
overall_roi = total_profit / total_spend

st.title("Marketing Campaign Performance + ROI Analysis")

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Marketing Spend", f"${total_spend:,.0f}")
kpi_2.metric("Revenue", f"${total_revenue:,.0f}")
kpi_3.metric("Blended CAC", f"${blended_cac:,.0f}")
kpi_4.metric("Gross-Profit ROI", f"{overall_roi * 100:.1f}%")

tab_channels, tab_campaigns, tab_budget, tab_data = st.tabs(["Channels", "Campaigns", "Budget Plan", "Data"])

with tab_channels:
    st.subheader("Channel Performance")
    st.dataframe(channel_summary_df, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        st.image(str(CHART_DIR / "roi_by_channel.png"))
    with right:
        st.image(str(CHART_DIR / "spend_vs_revenue_by_channel.png"))

with tab_campaigns:
    st.subheader("Campaign Decisions")
    decision = st.selectbox("Recommendation", ["All", "Scale", "Optimize", "Reduce / Test New Creative", "Stop"])
    filtered = campaign_summary_df if decision == "All" else campaign_summary_df[campaign_summary_df["recommendation"] == decision]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.image(str(CHART_DIR / "cac_by_campaign.png"))

with tab_budget:
    st.subheader("Recommended Budget Reallocation")
    st.dataframe(budget_plan_df, use_container_width=True, hide_index=True)
    st.image(str(CHART_DIR / "budget_reallocation.png"))

with tab_data:
    channels = ["All"] + sorted(campaign_data_df["channel"].unique().tolist())
    selected_channel = st.selectbox("Channel", channels)
    raw = campaign_data_df if selected_channel == "All" else campaign_data_df[campaign_data_df["channel"] == selected_channel]
    st.dataframe(raw, use_container_width=True, hide_index=True)

