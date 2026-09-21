# from llm_sdk import Small_LLM_Model
import parsing
import generate
from sys import argv


if __name__ == "__main__":
    f_path = ""
    p_path = ""
    output_path = ""
    if argv[1:]:
        if "--functions_definition" in argv:
            f_path = argv[argv.find("--functions_definition") + 1]
        if "--input" in argv:
            p_path = argv[argv.find("--functions_definition") + 1]
        if "--output" in argv:
            output_path = argv[argv.find("--functions_definition") + 1]
    test = parsing.Parse(f_path, p_path)
    test.prompt()
    test.function()
    gener = generate.Generator(test.all_functions, test.all_prompts)
    for prompt in test.all_prompts:
        print(gener.generate_function_name(prompt))
        print(gener.generate_param(prompt))
