# Marketing Campaign Performance + ROI Analysis

Portfolio-ready marketing analytics project that answers one practical business question:

**Which campaigns actually work, where should budget increase, and which campaigns should be stopped?**

## What This Project Covers

- Customer Acquisition Cost (CAC)
- Return on ad spend (ROAS)
- Marketing ROI based on gross profit
- Channel performance across Google, Facebook, Instagram, TikTok, Email, LinkedIn, and Affiliate
- Campaign-level stop / optimize / scale recommendations
- Budget reallocation scenario
- Executive-ready report and Streamlit dashboard

## Project Structure

```text
Marketing_Campaign_ROI_Analysis/
  app.py
  requirements.txt
  data/
    marketing_campaign_data.csv
  outputs/
    marketing_roi_report.md
    campaign_performance_summary.csv
    channel_performance_summary.csv
    budget_reallocation_plan.csv
    charts/
  src/
    marketing_campaign_roi_analysis.py
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset, charts, recommendations, and report:

```bash
python src/marketing_campaign_roi_analysis.py
```

Launch the dashboard:

```bash
streamlit run app.py --server.port 8502
```

## Business Story

The project uses a realistic synthetic dataset for an ecommerce company running paid and owned marketing campaigns. It calculates CAC, ROAS, ROI, conversion efficiency, and gross profit by campaign and channel, then recommends which campaigns deserve more budget and which should be paused.




