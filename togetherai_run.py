import os, asyncio
from together import AsyncTogether
from dotenv import load_dotenv
import aiofiles
from datetime import datetime
from typing import Any, AsyncIterator, Callable
import json
import pandas as pd
from tenacity import (
    retry,
    AsyncRetrying,
    stop_after_attempt,
    wait_random_exponential,
) 

load_dotenv()

seed = 42
temperature =  0
model_names = ["meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
model_name = model_names[0]

task_names = ["guess_value", "extract_value"]

task_name = task_names[0]

async def extract_q_request_id(df: pd.DataFrame):
    """
    Extracts tuples of ('Q', 'request_id') from rows where 'model_name' is None.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
    
    Returns:
        list: A list of tuples, each containing the 'Q' value and 'request_id' from rows where 'model_name' is None.
    """
    # Filter the DataFrame for rows where 'model_name' is None
    filtered_df = df[df[model_name].isnull()]
    
    # Select the 'Q' and 'request_id' columns and convert to list of tuples
    for row in filtered_df[['openai_message', 'request_id']].itertuples(index=False, name=None):
        await asyncio.sleep(0)
        yield row
    
    

async def save_results_to_file(filename: str, data: list):
    """
    Asynchronously save data to a file in JSON format.
    
    Args:
        filename (str): The name of the file where data will be saved.
        data (list): A list of processed items to save.
    """
    async with aiofiles.open(filename, 'w') as file:
        await file.write(json.dumps(data))

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def async_chat_completion(message):
    async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
    tasks = [
        async_client.chat.completions.create(
            model=model_name,
            messages=message,
        )
    ]
    responses = await asyncio.gather(*tasks)

    for response in responses:
        return response

async def process_iterator(it: AsyncIterator[Any], batch_size: int = 100):
    """
    Process items from an async iterator, saving results periodically to a file.
    
    Args:
        it (AsyncIterator[Any]): The async iterator providing data items.
        filename (str): The filename to save periodic results.
        batch_size (int): Number of results to collect before saving to a file.
    """
    results = []
    counter = 0
    
    async for item in it:
        message = json.loads(item[0])
        request_id = item[1]
        response = await async_chat_completion(message)
        results.append({"request_id": request_id,
                        model_name : response.choices[0].message.content})
        counter += 1
        
        if counter % batch_size == 0:
            filename = DATA_PATH / "togetherai_runs" / "checkpoints" / f"{counter}.json"
            await save_results_to_file(filename, results)
    
    # Save any remaining results
    if results:
        filename = DATA_PATH / "togetherai_runs" / "checkpoints" / f"finished.json"
        await save_results_to_file(filename, results)

    
async def main():
    if task_name == "guess_value":
        task_df = pd.read_parquet("dataset.parquet")

    if model_name not in task_df.columns:
        task_df[model_name] = None
    async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
    task_it = extract_q_request_id(task_df)
    
    await process_iterator(task_it)

if __name__ == "__main__":
    asyncio.run(main())
