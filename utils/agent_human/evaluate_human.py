import os
import re
import pandas as pd
from openpyxl import load_workbook
from core.environment import GridWorld

def extract_assignments_from_sheet(sheet):
    assignment = {}
    for row in range(25, 100):
        agent_cell = sheet[f"A{row}"]
        goal_cell = sheet[f"B{row}"]
        if agent_cell.value and isinstance(agent_cell.value, str) and agent_cell.value.strip().startswith("Agent"):
            agent_num = int(agent_cell.value.strip().split(" ")[1])
            goal_letter = str(goal_cell.value).strip().upper()
            assignment[agent_num] = goal_letter
        elif not agent_cell.value:
            break
    return assignment

def evaluate_human_xlsx(xlsx_path, configs_dir, output_csv):
    # Extract person index from file name
    basename = os.path.basename(xlsx_path)
    m = re.search(r"Person_(\d+)", basename)
    if not m:
        raise ValueError(f"Could not parse person number from file name: {basename}")
    person_num = int(m.group(1))
    print(f"Evaluating for Person {person_num}")

    wb = load_workbook(xlsx_path, data_only=True)
    sheets = wb.worksheets  # ordered as in file

    all_rows = []
    for sheet_idx, sheet in enumerate(sheets):
        # Each person’s sheet 1 = case_{(person_num-1)*10+1}.yaml, ..., sheet 10 = case_{person_num*10}.yaml
        case_num = (person_num - 1) * 10 + (sheet_idx + 1)
        case_name = f"case_{case_num}"
        config_path = os.path.join(configs_dir, f"{case_name}.yaml")
        if not os.path.exists(config_path):
            print(f"❌ Config missing for {case_name}")
            continue
        try:
            env = GridWorld(config_path)
            assignment = extract_assignments_from_sheet(sheet)
            cost = env.assignment_cost(assignment)
            all_rows.append({
                "Case": case_name,
                "Cost": cost,
                "Assignment": str(assignment)
            })
            print(f"✅ {case_name}: cost={cost} {assignment}")
        except Exception as e:
            print(f"❌ Error in {case_name}: {e}")

    # Write summary CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Results written to {output_csv}")
    else:
        print("⚠️ No results written.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True, help="Path to one human assignment XLSX file")
    parser.add_argument("--configs-dir", required=True, help="Directory with config YAML files")
    parser.add_argument("--output", default="human_person_summary.csv", help="CSV file for results")
    args = parser.parse_args()
    evaluate_human_xlsx(args.xlsx, args.configs_dir, args.output)
