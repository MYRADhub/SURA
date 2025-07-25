import os
import pandas as pd
from openpyxl import load_workbook
from core.environment import GridWorld

def extract_assignments_from_sheet(sheet):
    """
    Returns: dict[int, str] — mapping agent number to goal letter
    """
    assignment = {}
    for row in range(25, 100):  # scan rows 25 to 99
        agent_cell = sheet[f"A{row}"]
        goal_cell = sheet[f"B{row}"]
        if agent_cell.value and isinstance(agent_cell.value, str) and agent_cell.value.strip().startswith("Agent"):
            agent_num = int(agent_cell.value.strip().split(" ")[1])
            goal_letter = str(goal_cell.value).strip().upper()
            assignment[agent_num] = goal_letter
        elif not agent_cell.value:
            break  # stop if empty row
    return assignment

def process_workbook(filepath, configs_dir):
    """
    Extract from all sheets in a workbook
    Returns list of tuples: (case_name, assignment_dict, cost)
    """
    wb = load_workbook(filepath, data_only=True)
    results = []

    for sheetname in wb.sheetnames:
        sheet = wb[sheetname]
        case_name = sheetname.strip()

        config_path = os.path.join(configs_dir, f"{case_name}.yaml")
        if not os.path.exists(config_path):
            print(f"❌ Config missing for {case_name}")
            continue

        try:
            env = GridWorld(config_path)
            assignment = extract_assignments_from_sheet(sheet)
            cost = env.assignment_cost(assignment)
            results.append((case_name, assignment, cost))
        except Exception as e:
            print(f"❌ Error in {case_name}: {e}")
            continue

    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx-dir", required=True, help="Directory containing .xlsx human files")
    parser.add_argument("--configs-dir", required=True, help="Directory of YAML config files")
    parser.add_argument("--output", default="human_summary.csv", help="Path to save summary CSV")
    args = parser.parse_args()

    all_rows = []
    for filename in sorted(os.listdir(args.xlsx_dir)):
        if filename.endswith(".xlsx"):
            full_path = os.path.join(args.xlsx_dir, filename)
            print(f"📄 Processing: {filename}")
            results = process_workbook(full_path, args.configs_dir)
            for case, assignment, cost in results:
                row = {
                    "Case": case,
                    "Cost": cost,
                    "Assignment": str(assignment)
                }
                all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(args.output, index=False)
        print(f"\n✅ Saved summary to {args.output}")
    else:
        print("⚠️ No data extracted.")

if __name__ == "__main__":
    main()
