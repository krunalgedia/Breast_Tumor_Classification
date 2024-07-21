import yaml
import os
import sys
from modulefinder import ModuleFinder


def find_dependencies(script_path):
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"The script {script_path} does not exist.")

    finder = ModuleFinder()
    finder.run_script(script_path)

    dependencies = []
    for name, mod in finder.modules.items():
        if hasattr(mod, "__file__") and mod.__file__ is not None and not mod.is_builtin:
            dependencies.append(mod.__file__)

    return dependencies


def read_dvc_deps(dvc_file):
    if not os.path.exists(dvc_file):
        raise FileNotFoundError(f"The DVC file {dvc_file} does not exist.")

    with open(dvc_file, "r") as file:
        dvc_data = yaml.safe_load(file)

    return dvc_data["stages"]["run"]["deps"]


def main():
    main_script = "src/tumorClassify/pipeline/data_training_validation.py"
    dvc_file = "dvc.yaml"

    try:
        dependencies = find_dependencies(main_script)
        dvc_deps = read_dvc_deps(dvc_file)

        missing_deps = [dep for dep in dependencies if dep not in dvc_deps]

        if missing_deps:
            print("Missing dependencies:")
            for dep in missing_deps:
                print(f" - {dep}")
            return 1
        else:
            print("All dependencies are listed in the DVC file.")
            return 0

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return 1

    except Exception as e:
        print(f"An error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
