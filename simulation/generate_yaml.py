#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import yaml

input_file = "lm-evaluation-harness/lm_eval/tasks/multiplication/multiply.yaml"
with open(input_file, "r") as file:
    yaml_data = yaml.safe_load(file)  # Use safe_load to avoid security risks
    print("YAML data read from file:", yaml_data)

for i in range(10):
    yaml_data["task"] = f"multiply{i}"
    yaml_data["dataset_kwargs"]["data_files"]["validation"] = f"data/test{i}.json"

    output_file = f"lm-evaluation-harness/lm_eval/tasks/multiplication/multiply{i}.yaml"
    with open(output_file, "w") as file:
        yaml.dump(yaml_data, file, default_flow_style=False)
        print(f"Updated YAML written to {output_file}")
