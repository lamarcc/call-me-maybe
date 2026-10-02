from llm_sdk import Small_LLM_Model
import parsing
import generate
import argparse
import errors
import sys

parse = argparse.ArgumentParser()
parse.add_argument("--function_definition", type=str)
parse.add_argument("--input", type=str)
parse.add_argument("--output", type=str)

args = parse.parse_args()

if __name__ == "__main__":
    f_path = args.function_definition
    p_path = args.input
    o_path = args.output
    try:
        data = parsing.Parse(f_path, p_path)
        data.prompt()
        data.function()
    except errors.ParsingError as e:
        print(e)
        sys.exit(1)
    agent = Small_LLM_Model()
    gener = generate.Generator(data.all_functions, data.all_prompts)
    for prompt in data.all_prompts:
        # gener.build(prompt, data.all_functions)
        try:
            f = data.all_functions[
                agent.decode(gener.generate_function_name(prompt))
            ]
            print(f.name)
            p = gener.extract_param_value(f, prompt)
            for name, type in p.items():
                val = gener.get_value(f, prompt, name, type)
                print(val)
        except errors.GenerationError as e:
            print(f'{e} (prompt: "{prompt.prompt}")')
