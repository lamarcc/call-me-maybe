from llm_sdk import Small_LLM_Model
import parsing
import generate
import argparse
import errors
import sys


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
    for prompt in data.all_prompts:
        try:
            a = gener.build(prompt.prompt, data.all_functions)
            print(a)
        except errors.GenerationError as e:
            print(f'{e} (prompt: "{prompt.prompt}")')
    return 0


if __name__ == "__main__":
    sys.exit(main())
