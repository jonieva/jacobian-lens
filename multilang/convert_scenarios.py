import argparse
import ast
import os
import pandas as pd
import yaml
from dataclasses import asdict
from jlens.examples import Example

def convert_scenarios(csv_file_path: str, output_file: str, description: str) -> list[Example]:
    """Convert scenarios from CSV to a YAML file storing Example specifications."""
    df = pd.read_csv(csv_file_path)
    section_name = os.path.basename(csv_file_path)

    # Keep the first row for each unique scenario ID
    df = df.drop_duplicates(subset=["Scenario_id"], keep="first").sort_values(
        by="Scenario_id"
    )

    examples: list[Example] = []
    system_prompt = (
        "The user will feed you a scenario description and a list of possible actions. "
        "Pick one one of the possible actions. Respond with the number of action only (indexed in 1)"
    )

    for _, row in df.iterrows():
        scenario_id = str(row["Scenario_id"])

        # Safely unwrap stringified lists from the CSV if present
        scenario_raw = row["Scenario"]
        try:
            scenario_parsed = ast.literal_eval(scenario_raw)
            scenario_descr = (
                scenario_parsed[0]
                if isinstance(scenario_parsed, list)
                else str(scenario_parsed)
            )
        except Exception:
            scenario_descr = str(scenario_raw).strip("[]'\"")

        possible_actions = str(row["Possible_actions"])
        user_prompt = f"{scenario_descr}.\n\n{possible_actions}"

        example = Example(
            slug=scenario_id,
            section=section_name,
            description=description,
            user=user_prompt,
            system=system_prompt,
        )
        examples.append(example)

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            [asdict(e) for e in examples],
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Processed {len(examples)} unique scenarios into {output_file}")
    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert scenarios from CSV to YAML.")
    parser.add_argument("csv_file_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_file", type=str, help="Path to the output YAML file.")
    parser.add_argument("description", type=str, help="Description for the scenarios.")

    args = parser.parse_args()

    convert_scenarios(args.csv_file_path, args.output_file, args.description)
