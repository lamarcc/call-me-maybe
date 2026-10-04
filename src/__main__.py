from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pathlib import Path
from rich.progress import track
import argparse
import sys
import os
import json
import src.errors as errors
import src.parsing as parsing
import src.generate as generate


def create_output(dir_path: str, outputs: list[dict]) -> None:
    """Write generated function calls to a JSON file.

    Args:
        dir_path: Output file path. Defaults to
                  'data/output/functions_calls.json' if None or empty.
        outputs: List of dictionaries containing generated function calls.

    Raises:
        DirCreationError: If the output directory cannot be created due to
                         permission issues.
    """
    output_path = Path(dir_path) if dir_path else Path(
        'data/output/functions_calls.json'
    )
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(outputs, file, indent=2)
    except PermissionError:
        raise errors.DirCreationError(
            "Permission denied, can not create output directory"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace containing optional arguments for function definition file,
        input prompts file, and output file paths.
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("--function_definition", type=str)
    parse.add_argument("--input", type=str)
    parse.add_argument("--output", type=str)
    return parse.parse_args()


def main() -> int:
    """Main program execution function.

    Orchestrates the entire workflow: parse arguments, load data, generate
    function calls using an LLM, and write results to output file.

    Returns:
        0 if successful, 1 if a parsing error occurred.

    Handles:
        KeyboardInterrupt: Gracefully terminates on user interrupt.
        ParsingError: Catches and prints data validation errors.
        ProgramError: Catches and prints other program errors.
    """
    try:
        os.system("clear")
        args = parse_args()
        data = parsing.Parse(args.function_definition, args.input)
        data.prompt()
        data.function()
        agent = Small_LLM_Model()
        gener = generate.Generator(agent, data.all_functions)
        dicts = []
        for prompt in track(data.all_prompts, description="Processing..."):
            try:
                dicts.append(gener.build(prompt.prompt, data.all_functions))
            except errors.GenerationError as e:
                print(f'{e} <{prompt.prompt}>')
                dicts.append(
                    {
                        "prompt": json.dumps(prompt.prompt)[1:-1],
                        "name": "None",
                        "parameters": {"undefined": "None"}
                    }
                )
        create_output(args.output, dicts)
    except KeyboardInterrupt:
        print(errors.ProgramError.msg("Program interrupted"), end="")
    except errors.ParsingError as e:
        print(e)
        return 1
    except errors.ProgramError as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
