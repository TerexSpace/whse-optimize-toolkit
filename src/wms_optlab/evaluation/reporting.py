import pandas as pd
from typing import Dict

def generate_text_report(results: Dict, title: str = "Evaluation Report") -> str:
    """
    Generates a human-readable text report from a dictionary of results.
    """
    report = f"=== {title} ===\n\n"
    for key, value in results.items():
        if isinstance(value, dict):
            report += f"{key}:\n"
            for sub_key, sub_value in value.items():
                report += f"  - {sub_key}: {sub_value}\n"
        else:
            report += f"- {key}: {value}\n"
    return report

def generate_markdown_report(results_df: pd.DataFrame, title: str = "Evaluation Report") -> str:
    """
    Generates a markdown report from a pandas DataFrame.
    """
    report = f"# {title}\n\n"
    report += results_df.to_markdown(index=False)
    return report
