from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

if __name__ == "__main__":
    prompt = model.encode(input()).tolist()[0]
    print(prompt)
    try:
        for i in range(200):
            c = model.get_logits_from_input_ids(prompt)
            prompt.append(c.index(max(c)))
            print(model.decode(prompt))
    except KeyboardInterrupt:
        print("exit")
