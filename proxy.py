import asyncio
from typing import NamedTuple, List
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import heapq
from collections import defaultdict

# Upstream classification server
CLASSIFICATION_SERVER_URL = "http://localhost:8001/classify"

# Optimized configuration
MAX_BATCH_SIZE = 5
MAX_CONCURRENT_REQUESTS = 1
AGGRESSIVE_BATCH_TIMEOUT = 0.005  # Very short timeout for aggressive batching
RETRY_DELAY = 0.01

app = FastAPI(
    title="Classification Proxy - Optimized",
    description="High-performance proxy with mathematical optimization for batching"
)

class ProxyRequest(BaseModel):
    sequence: str

class ProxyResponse(BaseModel):
    result: str

class _QueueItem(NamedTuple):
    sequence: str
    future: asyncio.Future[str]
    timestamp: float
    length: int

# Single priority queue with intelligent batching
request_queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Metrics for adaptive optimization
batch_metrics = {
    'total_requests': 0,
    'total_batches': 0,
    'avg_batch_time': 0
}

@app.on_event("startup")
async def start_optimized_batcher():
    """Start the mathematically optimized batcher"""
    asyncio.create_task(_optimized_batcher())

@app.post("/proxy_classify", response_model=ProxyResponse)
async def proxy_classify(req: ProxyRequest):
    """
    Optimized endpoint that uses intelligent batching based on mathematical analysis
    """
    fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    item = _QueueItem(req.sequence, fut, time.time(), len(req.sequence))
    
    await request_queue.put(item)
    
    try:
        result = await asyncio.wait_for(fut, timeout=30.0)
        return ProxyResponse(result=result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Upstream classification timed out")

async def _optimized_batcher():
    """
    Mathematically optimized batcher that minimizes total processing time
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                # Get first item
                first_item = await request_queue.get()
                
                # Create optimal batch using mathematical strategy
                batch = await _create_optimal_batch(first_item)
                
                # Process batch
                await _process_batch_with_retries(client, batch)
                
            except Exception as e:
                print(f"Error in optimized batcher: {e}")
                await asyncio.sleep(0.001)

async def _create_optimal_batch(first_item: _QueueItem) -> List[_QueueItem]:
    """
    Create mathematically optimal batch to minimize processing time
    
    Strategy:
    1. Group items by similar lengths to minimize quadratic penalty
    2. Use aggressive timeouts during high-throughput periods
    3. Prioritize filling batches quickly over waiting for perfect matches
    """
    batch = [first_item]
    batch_start = time.time()
    
    # Length-based grouping for optimal batching
    target_length_range = _get_optimal_length_range(first_item.length)
    
    # Adaptive timeout based on queue size and current item length
    queue_size = request_queue.qsize()
    
    # More aggressive batching during high load
    if queue_size > 10:
        timeout = 0.001  # Ultra-fast batching under load
    elif queue_size > 5:
        timeout = 0.003
    elif first_item.length < 15:  # Small strings - batch quickly
        timeout = 0.005
    else:  # Large strings - slightly more patient
        timeout = 0.010
    
    # Collect items for optimal batch
    collected_items = []
    
    # Quickly drain available items
    while len(batch) < MAX_BATCH_SIZE and time.time() - batch_start < timeout:
        try:
            remaining_timeout = timeout - (time.time() - batch_start)
            if remaining_timeout <= 0:
                break
                
            item = await asyncio.wait_for(request_queue.get(), remaining_timeout)
            collected_items.append(item)
            
        except asyncio.TimeoutError:
            break
    
    # Intelligently select items for batch based on length optimization
    if collected_items:
        # Sort by length similarity to first item
        collected_items.sort(key=lambda x: abs(x.length - first_item.length))
        
        # Add items that don't significantly increase batch processing time
        current_max_length = first_item.length
        
        for item in collected_items:
            if len(batch) >= MAX_BATCH_SIZE:
                # Put remaining items back
                await request_queue.put(item)
            else:
                # Only add if it doesn't dramatically increase processing time
                new_max_length = max(current_max_length, item.length)
                time_increase = (new_max_length ** 2 - current_max_length ** 2) * 2e-3
                
                # Accept items that increase processing time by less than 50ms
                # or if batch is small
                if time_increase < 0.05 or len(batch) < 3:
                    batch.append(item)
                    current_max_length = new_max_length
                else:
                    # Put back items that would cause significant delay
                    await request_queue.put(item)
    
    return batch

def _get_optimal_length_range(length: int) -> tuple:
    """Get optimal length range for batching similar-sized items"""
    if length <= 10:
        return (1, 12)
    elif length <= 20:
        return (10, 25)
    else:
        return (20, 100)

async def _process_batch_with_retries(client: httpx.AsyncClient, batch: List[_QueueItem]):
    """
    Process batch with optimized retry logic and metrics tracking
    """
    async with request_semaphore:
        batch_start = time.time()
        sequences = [item.sequence for item in batch]
        futures = [item.future for item in batch]
        payload = {"sequences": sequences}
        
        # Calculate expected processing time for logging
        max_len = max(len(seq) for seq in sequences)
        expected_time = (max_len ** 2) * 2e-3
        
        success = False
        retry_count = 0
        max_retries = 3  # Reduced retries for faster failover
        
        while not success and retry_count < max_retries:
            try:
                resp = await client.post(CLASSIFICATION_SERVER_URL, json=payload)
                
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    
                    # Set results for all futures
                    for fut, result in zip(futures, results):
                        if not fut.done():
                            fut.set_result(result)
                    
                    # Update metrics
                    batch_time = time.time() - batch_start
                    _update_metrics(len(batch), batch_time, expected_time)
                    success = True
                    
                elif resp.status_code == 429:
                    # Quick exponential backoff
                    await asyncio.sleep(RETRY_DELAY * (1.5 ** retry_count))
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
                await asyncio.sleep(RETRY_DELAY * (1.5 ** retry_count))

def _update_metrics(batch_size: int, actual_time: float, expected_time: float):
    """Update performance metrics for adaptive optimization"""
    batch_metrics['total_requests'] += batch_size
    batch_metrics['total_batches'] += 1
    
    # Update running average of batch processing time
    if batch_metrics['avg_batch_time'] == 0:
        batch_metrics['avg_batch_time'] = actual_time
    else:
        # Exponential moving average
        alpha = 0.1
        batch_metrics['avg_batch_time'] = (alpha * actual_time + 
                                         (1 - alpha) * batch_metrics['avg_batch_time'])

@app.get("/metrics")
async def get_metrics():
    """Endpoint to monitor proxy performance"""
    return {
        "batch_metrics": batch_metrics,
        "queue_size": request_queue.qsize()
    }