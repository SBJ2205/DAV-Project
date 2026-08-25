"""
Script to generate a Word Document (.docx) guide for DAV Project Presentation.
"""
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DOCX = PROJECT_ROOT / "DAV_Project_Explanation_Guide.docx"


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def create_document():
    doc = docx.Document()

    # Page Margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Title
    title = doc.add_heading(level=0)
    run_title = title.add_run("Small Business Insolvency Early-Warning System")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(30, 58, 138)  # Deep blue

    sub = doc.add_paragraph()
    run_sub = sub.add_run("Data Analysis & Visualization (DAV) Minor Project — Comprehensive Presentation & Page-by-Page Guide")
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(100, 116, 139)

    doc.add_paragraph("―" * 55)

    # Section 1: Executive Summary & Project Purpose
    doc.add_heading("1. Project Concept & Big Picture (How to introduce the project)", level=1)
    p = doc.add_paragraph()
    p.add_run("What is this project about?\n").bold = True
    p.add_run(
        "It is a Machine Learning and Explainable AI (XAI) analytics system that predicts whether a company is at risk of "
        "bankruptcy based on its balance sheet and financial ratios, and visualizes exactly WHY that risk score was assigned."
    )

    p_why = doc.add_paragraph()
    p_why.add_run("Why did we build this?\n").bold = True
    p_why.add_run(
        "Traditional ML models are 'black boxes' — they give a prediction (e.g. 74% risk) but cannot explain which financial "
        "ratios caused that risk. Our system solves this by combining three layers:\n"
        "1. Predictive ML Models (Logistic Regression, Random Forest, XGBoost)\n"
        "2. Explainable AI (SHAP) to break down positive and negative risk factors\n"
        "3. A What-If Sensitivity Trajectory Engine to simulate managerial decisions."
    )

    doc.add_paragraph()

    # Section 2: Dataset Summary
    doc.add_heading("2. Dataset Understanding (Key facts for faculty)", level=1)
    table_data = [
        ["Attribute", "Value / Detail", "Why it matters in DAV"],
        ["Total Companies", "6,819 records", "Sufficiently large for statistical generalization."],
        ["Features", "95 financial ratios", "Covers Liquidity, Leverage, Profitability, and Cash Flow."],
        ["Target Variable", "Bankrupt? (0 = Healthy, 1 = Bankrupt)", "Binary classification problem."],
        ["Class Imbalance", "220 Bankrupt (3.23%) vs 6,599 Healthy (96.77%)", "Extreme imbalance! Raw Accuracy is misleading (96.8% by guessing 0)."],
        ["Data Nature", "Cross-Sectional (single time snapshot)", "No historical timelines; what-if is model sensitivity."],
    ]

    table = doc.add_table(rows=len(table_data), cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table_data):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = val
            if i == 0:
                set_cell_background(cell, "1E3A8A")
                for r in cell.paragraphs[0].runs:
                    r.font.color.rgb = RGBColor(255, 255, 255)
                    r.font.bold = True
            else:
                if i % 2 == 1:
                    set_cell_background(cell, "F8FAFC")

    doc.add_paragraph()

    # Section 3: Page-by-Page Breakdown
    doc.add_heading("3. Dashboard Page-by-Page Explanation", level=1)

    # Page 1
    doc.add_heading("Page 1: 📊 Overview & Model Intelligence", level=2)
    p1 = doc.add_paragraph()
    p1.add_run("What is on this page?\n").bold = True
    p1.add_run(
        "• 5 KPI Metric Cards showing dataset size (6,819), bankruptcy rate (3.23%), selected model (Random Forest), and test PR-AUC (0.4999).\n"
        "• Cohort Risk Distribution Histogram: Visualizes how many companies fall into Low (<30%), Moderate (30-60%), and High (>60%) risk categories.\n"
        "• Benchmark Model Comparison Table: Compares Logistic Regression, Random Forest, and XGBoost.\n"
        "• Methodological Transparency expander stating that data is cross-sectional and leakage-free."
    )

    p1_viva = doc.add_paragraph()
    p1_viva.add_run("What to say to Faculty / Group Mates:\n").bold = True
    p1_viva.add_run(
        "\"In our Overview page, we provide macroeconomic context on the entire company cohort. We explicitly show why "
        "PR-AUC (Precision-Recall AUC) is our primary metric instead of accuracy: because predicting all 0s gives 96.8% accuracy "
        "but catches zero bankruptcies. Random Forest won with the highest PR-AUC of 0.4999 and a 96.3% specificity.\""
    )

    doc.add_paragraph()

    # Page 2
    doc.add_heading("Page 2: 🔍 Company Insolvency Risk Assessment", level=2)
    p2 = doc.add_paragraph()
    p2.add_run("What is on this page?\n").bold = True
    p2.add_run(
        "• Company Selector: Lets the user pick any company record from the test set (or filter specifically by High, Moderate, or Low risk).\n"
        "• Prominent Risk Badge Card: Displays the model-predicted probability (e.g. 96.3%) with a color-coded visual progress bar.\n"
        "• Ground Truth Label Verification: Confirms whether the company actually went bankrupt in real life.\n"
        "• 4 Categorized Financial Tabs: Liquidity, Leverage, Profitability, and Operations — comparing the company's ratios directly against the cohort median."
    )

    p2_viva = doc.add_paragraph()
    p2_viva.add_run("What to say to Faculty / Group Mates:\n").bold = True
    p2_viva.add_run(
        "\"This is the individual company diagnosis tool. A bank manager or credit analyst can select any loan applicant, "
        "immediately see their predicted insolvency probability, and inspect 4 organized pillars of their financial statements "
        "to see where the company is deviating from healthy industry medians.\""
    )

    doc.add_paragraph()

    # Page 3
    doc.add_heading("Page 3: 🧠 Explainable AI & SHAP Insights", level=2)
    p3 = doc.add_paragraph()
    p3.add_run("What is on this page?\n").bold = True
    p3.add_run(
        "• Interactive SHAP Waterfall / Contribution Bar: Shows the exact mathematical contribution of each financial ratio to this company's score.\n"
        "• Red Drivers (Risk-Increasing): Financial factors that push the probability UP (e.g. high interest rates, high borrowing dependency).\n"
        "• Blue Buffers (Risk-Reducing): Financial factors that pull the probability DOWN (e.g. high cash flow, strong ROA).\n"
        "• Global SHAP Feature Importance: Ranks the top 15 most influential predictors across all 6,819 companies."
    )

    p3_viva = doc.add_paragraph()
    p3_viva.add_run("What to say to Faculty / Group Mates:\n").bold = True
    p3_viva.add_run(
        "\"Here is our core XAI contribution. Instead of giving an unexplainable score, SHAP uses game-theoretic Shapley values "
        "to calculate how much each individual ratio added or subtracted from the baseline risk. For Record #311, we can see that "
        "Continuous interest rate after tax and high debt pushed its risk to 96.3%.\""
    )

    doc.add_paragraph()

    # Page 4
    doc.add_heading("Page 4: 🔮 What-If Risk Trajectory & Sensitivity Engine", level=2)
    p4 = doc.add_paragraph()
    p4.add_run("What is on this page?\n").bold = True
    p4.add_run(
        "• Feature Selector: Pre-populated with the top SHAP drivers for this company.\n"
        "• Simulation Slider: Allows simulating changes between -50% and +50% in the chosen financial ratio.\n"
        "• Interactive Plotly Trajectory Curve: Plots the sensitivity curve, clearly marking the CURRENT state (red diamond) vs all hypothetical states (blue line).\n"
        "• Optimal Target Callout: Automatically computes the best simulated ratio value and tells the user how many percentage points of risk can be reduced."
    )

    p4_viva = doc.add_paragraph()
    p4_viva.add_run("What to say to Faculty / Group Mates:\n").bold = True
    p4_viva.add_run(
        "\"Because this is cross-sectional data, we cannot fabricate a fake time history. Instead, we built a What-If sensitivity engine. "
        "It freezes all other 93 ratios and runs real-time model inference as we adjust one variable. For instance, if this company "
        "reduces its borrowing dependency by 15%, our model recalculates that bankruptcy probability drops from 74% to 58%.\""
    )

    doc.add_paragraph()

    # Section 4: Quick Viva Questions & Model Answers
    doc.add_heading("4. Likely Faculty Viva Questions & Perfect Answers", level=1)

    qa_list = [
        ("Q1: Why did you choose Random Forest over Logistic Regression and XGBoost?",
         "Random Forest achieved the highest PR-AUC (0.4999) on the test set while keeping false positives low (96.3% specificity). Logistic regression had high recall (81.8%) but too many false alarms (precision only 18.5%). Random Forest struck the best balance."),
        ("Q2: How did you prevent data leakage during preprocessing?",
         "All transformations (median imputation and standard scaling) were fitted strictly on the 80% training partition (X_train). The 20% test partition was only transformed and never seen during fitting. Constant features were dynamically removed."),
        ("Q3: Why not just use Accuracy as your evaluation metric?",
         "Because only 3.23% of companies in the dataset are bankrupt. A dummy model that predicts 'Healthy' for every single company achieves 96.77% accuracy while completely failing at its job. That is why PR-AUC, Recall, and F1 are the scientifically valid metrics."),
        ("Q4: What is SHAP and why is it better than standard feature importance?",
         "Standard feature importance only gives global importance for the whole model. SHAP gives local explanations for every single individual company, showing both the direction (increasing or reducing risk) and exact magnitude of impact."),
    ]

    for q, a in qa_list:
        p_qa = doc.add_paragraph()
        p_qa.add_run(f"{q}\n").bold = True
        p_qa.add_run(f"Answer: {a}")

    # Save
    doc.save(OUTPUT_DOCX)
    print(f"Document saved to {OUTPUT_DOCX}")


if __name__ == "__main__":
    create_document()
