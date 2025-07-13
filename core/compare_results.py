import csv
import argparse
from collections import defaultdict

def load_results(file_path, is_multi_trial=False):
    """
    Load results from a CSV file.
    If is_multi_trial is True, average results over multiple trials per case.
    Returns a dict: case_name -> {"steps": avg, "opt": avg, "fail": total, "collisions": avg}
    """
    results = defaultdict(list)

    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = row["Case"]
            steps = int(row["Steps"])
            opt = int(row["Optimal"])
            failed = int(row["Failed"])
            collisions = int(row["Collisions"])
            results[case].append((steps, opt, failed, collisions))

    aggregated = {}
    for case, trials in results.items():
        n = len(trials)
        avg_steps = sum(t[0] for t in trials) / n
        avg_opt = sum(t[1] for t in trials) / n
        total_failed = sum(t[2] for t in trials)
        avg_collisions = sum(t[3] for t in trials) / n
        aggregated[case] = {
            "steps": avg_steps,
            "opt": avg_opt,
            "fail": total_failed,
            "collisions": avg_collisions
        }

    return aggregated

def compare(agent1_results, agent2_results, label1="Agent 1", label2="Agent 2"):
    shared_cases = sorted(set(agent1_results) & set(agent2_results))
    if not shared_cases:
        print("No overlapping cases to compare.")
        return

    stats = {
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "failures_agent1": 0,
        "failures_agent2": 0,
        "total_cases": len(shared_cases),
        "total_collisions_agent1": 0,
        "total_collisions_agent2": 0,
        "total_steps_agent1": 0,
        "total_steps_agent2": 0,
    }

    print(f"\nComparing {len(shared_cases)} shared cases:")
    for case in shared_cases:
        a1 = agent1_results[case]
        a2 = agent2_results[case]

        s1 = a1["steps"]
        s2 = a2["steps"]
        stats["total_steps_agent1"] += s1
        stats["total_steps_agent2"] += s2

        stats["total_collisions_agent1"] += a1["collisions"]
        stats["total_collisions_agent2"] += a2["collisions"]

        stats["failures_agent1"] += a1["fail"]
        stats["failures_agent2"] += a2["fail"]

        if s1 < s2:
            stats["wins"] += 1
        elif s1 > s2:
            stats["losses"] += 1
        else:
            stats["ties"] += 1

    print(f"\n📊 Comparison Summary ({label1} vs {label2}):")
    print(f"- Total Cases Compared: {stats['total_cases']}")
    print(f"- {label1} Wins (Fewer Steps): {stats['wins']}")
    print(f"- {label2} Wins (Fewer Steps): {stats['losses']}")
    print(f"- Ties: {stats['ties']}")
    print(f"- Avg Steps: {label1}: {stats['total_steps_agent1'] / stats['total_cases']:.2f}, {label2}: {stats['total_steps_agent2'] / stats['total_cases']:.2f}")
    print(f"- Avg Collisions: {label1}: {stats['total_collisions_agent1'] / stats['total_cases']:.2f}, {label2}: {stats['total_collisions_agent2'] / stats['total_cases']:.2f}")
    print(f"- Failures: {label1}: {stats['failures_agent1']}, {label2}: {stats['failures_agent2']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two agent result CSVs by case.")
    parser.add_argument("--agent1", required=True, help="Path to first agent's CSV file")
    parser.add_argument("--agent2", required=True, help="Path to second agent's CSV file")
    parser.add_argument("--label1", default="Agent 1", help="Label for first agent")
    parser.add_argument("--label2", default="Agent 2", help="Label for second agent")
    parser.add_argument("--multi-trial", action="store_true", help="Average over multiple trials per case")
    args = parser.parse_args()

    agent1_data = load_results(args.agent1, is_multi_trial=args.multi_trial)
    agent2_data = load_results(args.agent2, is_multi_trial=args.multi_trial)

    compare(agent1_data, agent2_data, label1=args.label1, label2=args.label2)
