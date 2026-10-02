from llm_sdk import Small_LLM_Model
import parsing
import generate
import argparse
import errors
import sys
import json


def parse_args() -> argparse.Namespace:
    parse = argparse.ArgumentParser()
    parse.add_argument("--function_definition", type=str)
    parse.add_argument("--input", type=str)
    parse.add_argument("--output", type=str)
    return parse.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = parsing.Parse(args.function_definition, args.input)
        data.prompt()
        data.function()
    except errors.ParsingError as e:
        print(e)
        return 1
    agent = Small_LLM_Model()
    gener = generate.Generator(agent, data.all_functions)
    dicts = []
    if not args.output:
        args.output = 'data/output/output.json'
    try:
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
        print(dicts)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(dicts, file, indent=2)
    except KeyboardInterrupt as e:
        print(errors.ProgramError.message(e, "Program interrupted"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
