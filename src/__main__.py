from llm_sdk import Small_LLM_Model
from pathlib import Path
import argparse
import sys
import json
import errors
import parsing
import generate


def create_output(dir_path: str, outputs: list[dict]) -> None:
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
    parse = argparse.ArgumentParser()
    parse.add_argument("--function_definition", type=str)
    parse.add_argument("--input", type=str)
    parse.add_argument("--output", type=str)
    return parse.parse_args()


def main() -> int:
    try:
        args = parse_args()
        data = parsing.Parse(args.function_definition, args.input)
        data.prompt()
        data.function()
        agent = Small_LLM_Model()
        gener = generate.Generator(agent, data.all_functions)
        dicts = []
        for prompt in data.all_prompts:
            try:
                dicts.append(gener.build(prompt.prompt, data.all_functions))
            except errors.GenerationError as e:
                print(f'{e} (prompt: "{prompt.prompt}")')
                dicts.append(
                    {
                        "prompt": prompt,
                        "name": "None",
                        "parameters": {"None": "None"}
                    }
                )
        create_output(args.output, dicts)
    except KeyboardInterrupt as e:
        print(errors.ProgramError.msg(e, "Program interrupted"), end="")
    except errors.ParsingError as e:
        print(e)
        return 1
    except errors.ProgramError as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
