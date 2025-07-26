import os
import pandas as pd
from openpyxl import load_workbook
from core.environment import GridWorld

def extract_assignments_from_sheet(sheet):
    assignment = {}
    for row in range(25, 100):  # Scan from row 25 down
        agent_cell = sheet[f"A{row}"]
        goal_cell = sheet[f"B{row}"]
        if agent_cell.value and isinstance(agent_cell.value, str) and agent_cell.value.strip().startswith("Agent"):
            agent_num = int(agent_cell.value.strip().split(" ")[1])
            goal_letter = str(goal_cell.value).strip().upper()
            assignment[agent_num] = goal_letter
        elif not agent_cell.value:
            break
    return assignment

def process_all_files(xlsx_dir, configs_dir, output):
    # Get all xlsx files, sort for deterministic ordering
    xlsx_files = sorted([f for f in os.listdir(xlsx_dir) if f.endswith(".xlsx")])
    total_cases = 0
    all_rows = []
    case_counter = 1
    # Sort the files in numerical order ("Person_10_louis" comes after "Person_2_jad")
    if not xlsx_files:
        print("No XLSX files found in the specified directory.")
        return
    xlsx_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Extract number from filename
    print(f"Listing {len(xlsx_files)} XLSX files to process...")
    print(xlsx_files)

    for filename in xlsx_files:
        wb = load_workbook(os.path.join(xlsx_dir, filename), data_only=True)
        sheets = wb.worksheets  # list in order
        print(len(sheets), filename)
        for sheet in sheets:
            # Map sheet index (within file) to case_X.yaml
            case_name = f"case_{case_counter}"
            config_path = os.path.join(configs_dir, f"{case_name}.yaml")
            if not os.path.exists(config_path):
                print(f"❌ Config missing for {case_name}")
                case_counter += 1
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
            except Exception as e:
                print(f"❌ Error in {case_name}: {e}")
            case_counter += 1
            total_cases += 1

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output, index=False)
        print(f"\n✅ Saved summary for {total_cases} sheets to {output}")
    else:
        print("⚠️ No data extracted.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx-dir", required=True, help="Directory with all XLSX files")
    parser.add_argument("--configs-dir", required=True, help="Directory with config YAML files")
    parser.add_argument("--output", default="human_summary.csv", help="CSV file for output")
    args = parser.parse_args()
    process_all_files(args.xlsx_dir, args.configs_dir, args.output)
