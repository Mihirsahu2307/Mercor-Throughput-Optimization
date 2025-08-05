import asyncio
from typing import NamedTuple
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

# Upstream classification server
CLASSIFICATION_SERVER_URL = "http://localhost:8001/classify"

# Optimized batching configuration
MAX_BATCH_SIZE = 5                        # max sequences per batch
SMALL_THRESHOLD = 12                      # length threshold for small vs large
BATCH_WAIT_SMALL = 0.02                   # reduced wait time for small sequences
BATCH_WAIT_LARGE = 0.15                   # reduced wait time for large sequences  
RETRY_DELAY = 0.05                        # reduced retry delay
MAX_CONCURRENT_REQUESTS = 1               # server limitation

app = FastAPI(
    title="Classification Proxy",
    description="Optimized proxy server with intelligent, size-aware batching and concurrent processing"
)

class ProxyRequest(BaseModel):
    sequence: str

class ProxyResponse(BaseModel):
    result: str

class _QueueItem(NamedTuple):
    sequence: str
    future: asyncio.Future[str]
    timestamp: float

# Separate queues for small and large sequences
small_queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
large_queue: asyncio.Queue[_QueueItem] = asyncio.Queue()

# Semaphore to control concurrent requests to the classification server
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

@app.on_event("startup")
async def start_batcher():
    # Launch multiple background batching workers for better concurrency
    asyncio.create_task(_small_batcher())
    asyncio.create_task(_large_batcher())

@app.post("/proxy_classify", response_model=ProxyResponse)
async def proxy_classify(req: ProxyRequest):
    """
    Receives a single sequence, enqueues it for batching,
    and waits for the result from the background worker.
    """
    fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    item = _QueueItem(req.sequence, fut, time.time())
    
    # Route into small or large bucket based on sequence length
    if len(req.sequence) < SMALL_THRESHOLD:
        await small_queue.put(item)
    else:
        await large_queue.put(item)
    
    try:
        result = await asyncio.wait_for(fut, timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Upstream classification timed out")
    
    return ProxyResponse(result=result)

async def _small_batcher():
    """
    Dedicated batcher for small sequences with aggressive batching.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                # Wait for at least one item
                first_item = await small_queue.get()
                batch = [first_item]
                
                # Try to fill the batch quickly
                batch_start = time.time()
                while len(batch) < MAX_BATCH_SIZE:
                    elapsed = time.time() - batch_start
                    timeout = BATCH_WAIT_SMALL - elapsed
                    if timeout <= 0:
                        break
                    
                    try:
                        item = await asyncio.wait_for(small_queue.get(), timeout)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                # Process the batch
                await _process_batch(client, batch)
                
            except Exception as e:
                print(f"Error in small batcher: {e}")
                await asyncio.sleep(0.01)

async def _large_batcher():
    """
    Dedicated batcher for large sequences with more patient batching.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                # Wait for at least one item
                first_item = await large_queue.get()
                batch = [first_item]
                
                # Try to batch large items together, but be more patient
                batch_start = time.time()
                while len(batch) < MAX_BATCH_SIZE:
                    elapsed = time.time() - batch_start
                    timeout = BATCH_WAIT_LARGE - elapsed
                    if timeout <= 0:
                        break
                    
                    try:
                        item = await asyncio.wait_for(large_queue.get(), timeout)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                # Process the batch
                await _process_batch(client, batch)
                
            except Exception as e:
                print(f"Error in large batcher: {e}")
                await asyncio.sleep(0.01)

async def _process_batch(client: httpx.AsyncClient, batch: list[_QueueItem]):
    """
    Process a batch of items by sending them to the classification server.
    Uses semaphore to respect server's single-request limitation.
    """
    async with request_semaphore:
        sequences = [item.sequence for item in batch]
        futures = [item.future for item in batch]
        payload = {"sequences": sequences}
        
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries:
            try:
                resp = await client.post(CLASSIFICATION_SERVER_URL, json=payload)
                
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    # Set results for all futures
                    for fut, label in zip(futures, results):
                        if not fut.done():
                            fut.set_result(label)
                    return
                    
                elif resp.status_code == 429:
                    # Exponential backoff for rate limiting
                    await asyncio.sleep(RETRY_DELAY * (2 ** retry_count))
                    retry_count += 1
                    
                else:
                    resp.raise_for_status()
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    # Set exceptions for all futures
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(e)
                    return
                    
                await asyncio.sleep(RETRY_DELAY * (2 ** retry_count))