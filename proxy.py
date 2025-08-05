import asyncio
from typing import NamedTuple, List, Dict
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

# Upstream classification server
CLASSIFICATION_SERVER_URL = "http://localhost:8001/classify"

# Advanced configuration
MAX_BATCH_SIZE = 5
MAX_CONCURRENT_REQUESTS = 1
ULTRA_FAST_TIMEOUT = 0.001  # For high-frequency small requests
FAST_TIMEOUT = 0.003
MEDIUM_TIMEOUT = 0.008
SLOW_TIMEOUT = 0.015

app = FastAPI(
    title="Classification Proxy - Advanced Optimized",
    description="Multi-queue, priority-based proxy with predictive batching"
)

class ProxyRequest(BaseModel):
    sequence: str

class ProxyResponse(BaseModel):
    result: str

class Priority(Enum):
    URGENT = 1      # Very small strings that can batch quickly
    HIGH = 2        # Small-medium strings
    MEDIUM = 3      # Medium strings
    LOW = 4         # Large strings

@dataclass
class QueueItem:
    sequence: str
    future: asyncio.Future[str]
    timestamp: float
    length: int
    priority: Priority
    
    def __lt__(self, other):
        return self.priority.value < other.priority.value

# Multiple specialized queues for different request types
class MultiQueueSystem:
    def __init__(self):
        # Length-based queues for optimal batching
        self.tiny_queue = asyncio.Queue()      # 1-8 chars
        self.small_queue = asyncio.Queue()     # 9-12 chars  
        self.medium_queue = asyncio.Queue()    # 13-18 chars
        self.large_queue = asyncio.Queue()     # 19-25 chars
        self.xlarge_queue = asyncio.Queue()    # 26+ chars
        
        # Priority queue for urgent processing
        self.priority_heap = []
        self.priority_lock = asyncio.Lock()
        
        # Batch prediction based on recent patterns
        self.recent_patterns = deque(maxlen=50)
        self.batch_efficiency_history = {}
        
    async def enqueue(self, item: QueueItem):
        """Smart enqueueing based on length and current system state"""
        length = item.length
        
        # Determine priority based on length and system load
        if length <= 8:
            item.priority = Priority.URGENT
            await self.tiny_queue.put(item)
        elif length <= 12:
            item.priority = Priority.HIGH
            await self.small_queue.put(item)
        elif length <= 18:
            item.priority = Priority.MEDIUM
            await self.medium_queue.put(item)
        elif length <= 25:
            item.priority = Priority.LOW
            await self.large_queue.put(item)
        else:
            item.priority = Priority.LOW
            await self.xlarge_queue.put(item)
            
        # Track patterns for predictive batching
        self.recent_patterns.append((length, time.time()))
    
    async def get_optimal_batch(self) -> List[QueueItem]:
        """Get the most optimal batch based on current system state"""
        # Check for urgent tiny requests first (they batch extremely efficiently)
        if not self.tiny_queue.empty():
            return await self._batch_from_queue(self.tiny_queue, ULTRA_FAST_TIMEOUT)
        
        # Predict best queue based on recent patterns and efficiency
        best_queue = await self._predict_best_queue()
        if best_queue:
            return await self._batch_from_queue(best_queue[0], best_queue[1])
            
        # Fallback to first available queue
        for queue, timeout in [
            (self.small_queue, FAST_TIMEOUT),
            (self.medium_queue, MEDIUM_TIMEOUT), 
            (self.large_queue, SLOW_TIMEOUT),
            (self.xlarge_queue, SLOW_TIMEOUT)
        ]:
            if not queue.empty():
                return await self._batch_from_queue(queue, timeout)
        
        return []
    
    async def _predict_best_queue(self):
        """Predictive algorithm to choose optimal queue"""
        queue_priorities = []
        
        # Check queue sizes and calculate efficiency scores
        queues_info = [
            (self.small_queue, FAST_TIMEOUT, 10, "small"),
            (self.medium_queue, MEDIUM_TIMEOUT, 15, "medium"),
            (self.large_queue, SLOW_TIMEOUT, 22, "large"),
            (self.xlarge_queue, SLOW_TIMEOUT, 30, "xlarge")
        ]
        
        for queue, timeout, avg_length, name in queues_info:
            size = queue.qsize()
            if size > 0:
                # Calculate efficiency: requests per second potential
                batch_time = (avg_length ** 2) * 2e-3 + timeout
                efficiency = min(size, MAX_BATCH_SIZE) / batch_time
                
                # Bonus for larger batches and recent successful patterns
                batch_bonus = min(size, MAX_BATCH_SIZE) * 0.1
                pattern_bonus = self.batch_efficiency_history.get(name, 1.0)
                
                total_score = efficiency + batch_bonus + pattern_bonus
                queue_priorities.append((total_score, queue, timeout, name))
        
        if queue_priorities:
            queue_priorities.sort(reverse=True)
            best = queue_priorities[0]
            return (best[1], best[2])  # queue, timeout
        
        return None
    
    async def _batch_from_queue(self, queue: asyncio.Queue, timeout: float) -> List[QueueItem]:
        """Extract optimal batch from specific queue"""
        if queue.empty():
            return []
            
        try:
            # Get first item
            first_item = await asyncio.wait_for(queue.get(), timeout=0.001)
            batch = [first_item]
            batch_start = time.time()
            
            # Aggressively collect more items
            while len(batch) < MAX_BATCH_SIZE:
                remaining_time = timeout - (time.time() - batch_start)
                if remaining_time <= 0:
                    break
                    
                try:
                    item = await asyncio.wait_for(queue.get(), remaining_time)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            return batch
            
        except asyncio.TimeoutError:
            return []

# Global queue system
queue_system = MultiQueueSystem()
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Performance monitoring
performance_stats = {
    'batches_processed': 0,
    'total_items': 0,
    'avg_batch_size': 0,
    'processing_times': deque(maxlen=100)
}

@app.on_event("startup")
async def start_advanced_batchers():
    """Start multiple specialized batching workers"""
    # Multiple workers for maximum concurrency
    for i in range(3):  # 3 workers for different priorities
        asyncio.create_task(_advanced_batcher(worker_id=i))
    
    # Dedicated fast-track worker for tiny requests
    asyncio.create_task(_fast_track_batcher())
    
    # Background optimizer
    asyncio.create_task(_performance_optimizer())

@app.post("/proxy_classify", response_model=ProxyResponse)
async def proxy_classify(req: ProxyRequest):
    """Ultra-optimized endpoint with predictive batching"""
    fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    item = QueueItem(
        sequence=req.sequence,
        future=fut,
        timestamp=time.time(),
        length=len(req.sequence),
        priority=Priority.MEDIUM  # Will be updated in enqueue
    )
    
    await queue_system.enqueue(item)
    
    try:
        result = await asyncio.wait_for(fut, timeout=25.0)
        return ProxyResponse(result=result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")

async def _fast_track_batcher():
    """Dedicated worker for tiny requests - ultra-fast processing"""
    async with httpx.AsyncClient(timeout=20.0) as client:
        while True:
            try:
                if queue_system.tiny_queue.qsize() > 0:
                    batch = await queue_system._batch_from_queue(
                        queue_system.tiny_queue, 
                        ULTRA_FAST_TIMEOUT
                    )
                    if batch:
                        await _process_batch_optimally(client, batch, "fast_track")
                else:
                    await asyncio.sleep(0.001)  # Very short sleep
            except Exception as e:
                print(f"Fast track batcher error: {e}")
                await asyncio.sleep(0.001)

async def _advanced_batcher(worker_id: int):
    """Advanced batching worker with predictive optimization"""
    async with httpx.AsyncClient(timeout=20.0) as client:
        while True:
            try:
                # Get optimal batch using prediction algorithm
                batch = await queue_system.get_optimal_batch()
                
                if batch:
                    await _process_batch_optimally(client, batch, f"worker_{worker_id}")
                else:
                    # Brief sleep when no work available
                    await asyncio.sleep(0.002)
                    
            except Exception as e:
                print(f"Advanced batcher {worker_id} error: {e}")
                await asyncio.sleep(0.001)

async def _process_batch_optimally(client: httpx.AsyncClient, batch: List[QueueItem], worker_name: str):
    """Optimized batch processing with performance tracking"""
    if not batch:
        return
        
    async with request_semaphore:
        start_time = time.time()
        sequences = [item.sequence for item in batch]
        futures = [item.future for item in batch]
        
        # Pre-calculate expected time for optimization
        max_length = max(len(seq) for seq in sequences)
        expected_time = (max_length ** 2) * 2e-3
        
        payload = {"sequences": sequences}
        success = False
        attempts = 0
        max_attempts = 2  # Fast failover
        
        while not success and attempts < max_attempts:
            try:
                # Use shorter timeout for faster failover
                response = await client.post(
                    CLASSIFICATION_SERVER_URL, 
                    json=payload,
                    timeout=15.0
                )
                
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    
                    # Set results
                    for fut, result in zip(futures, results):
                        if not fut.done():
                            fut.set_result(result)
                    
                    success = True
                    
                    # Update performance tracking
                    processing_time = time.time() - start_time
                    _update_performance_stats(len(batch), processing_time, expected_time, worker_name)
                    
                elif response.status_code == 429:
                    # Immediate retry with minimal delay
                    await asyncio.sleep(0.005 * (attempts + 1))
                    attempts += 1
                else:
                    response.raise_for_status()
                    
            except Exception as e:
                attempts += 1
                if attempts >= max_attempts:
                    # Set exceptions
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(e)
                    return
                await asyncio.sleep(0.005)

def _update_performance_stats(batch_size: int, processing_time: float, expected_time: float, worker_name: str):
    """Update performance statistics for optimization"""
    performance_stats['batches_processed'] += 1
    performance_stats['total_items'] += batch_size
    performance_stats['processing_times'].append(processing_time)
    
    # Update running average
    total_batches = performance_stats['batches_processed']
    performance_stats['avg_batch_size'] = performance_stats['total_items'] / total_batches
    
    # Update queue efficiency history
    efficiency = batch_size / (processing_time + 0.001)  # Avoid division by zero
    
    # Store efficiency by batch characteristics for future prediction
    if batch_size >= 4:
        queue_system.batch_efficiency_history[worker_name] = efficiency * 0.1 + \
            queue_system.batch_efficiency_history.get(worker_name, 1.0) * 0.9

async def _performance_optimizer():
    """Background task to optimize performance based on patterns"""
    while True:
        await asyncio.sleep(0.5)  # Check every 500ms
        
        try:
            # Adjust timeouts based on recent performance
            recent_times = list(performance_stats['processing_times'])[-10:]
            if recent_times:
                avg_time = sum(recent_times) / len(recent_times)
                
                # If processing is fast, be more aggressive with timeouts
                if avg_time < 0.1:
                    # System is fast, reduce timeouts
                    global FAST_TIMEOUT, MEDIUM_TIMEOUT
                    FAST_TIMEOUT = max(0.001, FAST_TIMEOUT * 0.95)
                    MEDIUM_TIMEOUT = max(0.003, MEDIUM_TIMEOUT * 0.95)
                elif avg_time > 0.5:
                    # System is slow, increase timeouts slightly
                    FAST_TIMEOUT = min(0.01, FAST_TIMEOUT * 1.05)
                    MEDIUM_TIMEOUT = min(0.02, MEDIUM_TIMEOUT * 1.05)
                    
        except Exception as e:
            print(f"Performance optimizer error: {e}")

@app.get("/stats")
async def get_performance_stats():
    """Get detailed performance statistics"""
    return {
        "performance_stats": performance_stats,
        "queue_sizes": {
            "tiny": queue_system.tiny_queue.qsize(),
            "small": queue_system.small_queue.qsize(),
            "medium": queue_system.medium_queue.qsize(),
            "large": queue_system.large_queue.qsize(),
            "xlarge": queue_system.xlarge_queue.qsize()
        },
        "efficiency_history": queue_system.batch_efficiency_history
    }