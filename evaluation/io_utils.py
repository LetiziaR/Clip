import json


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def save_generations_jsonl(path, predictions, references):
    with open(path, "w", encoding="utf-8") as fp:
        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            item = {
                "index": idx,
                "prediction": pred,
                "reference": ref,
            }
            fp.write(json.dumps(item, ensure_ascii=True) + "\n")
