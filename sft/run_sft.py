import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def preprocess_function(args):
    # load dataset
    train_dataset = load_dataset("json", data_files=args.dataset)
    prefix = "Based on the QA history, rewrite the last quetion:"
    inputs = [prefix + data["input"] for data in train_dataset["train"]]
    targets = [data["output"] for data in train_dataset["train"]]
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True, is_split_into_words=True)
    print(model_inputs[0], len(model_inputs))
    return model_inputs

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="./dataset/CoQAR/data/new_coqar_train.json")

    args = parser.parse_args()

    preprocess_function(args)

if __name__ == "__main__":
    main()