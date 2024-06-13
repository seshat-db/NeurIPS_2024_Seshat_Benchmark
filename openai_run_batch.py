from openai import OpenAI
import os
from dotenv import load_dotenv
import polars as pl
from tqdm import tqdm
import datetime
import json
from pathlib import Path

load_dotenv()


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
seed = 42
model_names = ["gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0125",]
model_name = model_names[1]

outputdir = Path("open_ai_requests")
task_df = pl.read_parquet("dataset.parquet")
tasks = list()
batch_num = 0

for ix, q in tqdm(enumerate(task_df.iter_rows(named=True))):
    request_id = q["request_id"]
    message = json.loads(q["openai_message"])
    task = {
        "custom_id": f"{request_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "seed": 42,
            "temperature": 0,
            "max_tokens": 1,
            "messages": message,
        },
    }

    tasks.append(task)

    if ix % 10000 == 0 and ix > 0:
        batch_save_path = (
            outputdir
            / f'guess_value_Batch-{batch_num}_{model_name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.jsonl'
        )
        with open(batch_save_path, "w") as handle:
            for entry in tasks:
                json.dump(entry, handle)
                handle.write("\n")
        batch_file = client.files.create(
            file=open(batch_save_path, "rb"), purpose="batch"
        )
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        tasks = list()
        batch_num += 1


batch_save_path = (
    outputdir
    / f'guess_value_Batch-{batch_num}_{model_name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.jsonl'
)
with open(batch_save_path, "w") as handle:
    for entry in tasks:
        json.dump(entry, handle)
        handle.write("\n")
batch_file = client.files.create(
    file=open(batch_save_path, "rb"), purpose="batch"
)
batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
