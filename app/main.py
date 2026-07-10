"""Entry point and orchestration for the Video Captioning Agent.

Architecture: Producer-Consumer with Model Pools
- Stage 1 (Producers): Multimodal models generate descriptions from video/keyframes
- Stage 2 (Consumers): Text models generate styled captions from descriptions
- Queues: asyncio.Queue connects the two stages
"""

import asyncio
import glob
import logging
import os
import sys
import time
from dataclasses import dataclass

from app.config import (
    FALLBACK_CAPTION,
    INPUT_PATH,
    OUTPUT_PATH,
    RUNTIME_TIMEOUT,
)
from app.downloader import download_video
from app.keyframe import extract_keyframes
from app.models import Task, TaskResult
from app.output_writer import write_results
from app.providers import (
    GenerationError,
    UploadError,
    build_caption_providers,
    build_description_providers,
    load_model_config,
)
from app.providers.base import CaptionProvider, DescriptionProvider
from app.task_parser import parse_tasks
from app.sanitize import sanitize_caption

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DescriptionItem:
    """Item passed from Stage 1 producers to Stage 2 consumers."""
    task: Task
    description: str


# ---------- Stage 1: Description Producers ----------

async def produce_description(
    task: Task,
    description_pool: asyncio.Queue,
    description_queue: asyncio.Queue,
    start_time: float,
) -> bool:
    """Download video, extract keyframes, generate description, push to queue.

    Args:
        task: The task to process.
        description_pool: Queue of available DescriptionProvider instances.
        description_queue: Queue to push completed descriptions into.
        start_time: Monotonic start time for timeout checking.

    Returns:
        True if description was produced, False if task was skipped.
    """
    # Check timeout
    if time.monotonic() - start_time > RUNTIME_TIMEOUT:
        logger.warning("Timeout reached, skipping task %s", task.task_id)
        return False

    # Step 1: Download video
    try:
        video_path = await download_video(task.task_id, task.video_url)
        logger.info("Downloaded video for task %s", task.task_id)
    except Exception as exc:
        logger.error("Download failed for task %s: %s", task.task_id, exc)
        # Push fallback directly to description queue
        await description_queue.put(None)
        return False

    # Step 2: Get a description provider from the pool
    provider: DescriptionProvider = await description_pool.get()

    try:
        # Step 3: Try upload, fall back to keyframes
        video_ref: str | None = None
        keyframe_paths: list[str] | None = None

        try:
            video_ref = await provider.upload_video(video_path)
            logger.info("Uploaded video for task %s", task.task_id)
        except (UploadError, Exception) as exc:
            logger.warning(
                "Upload failed for task %s, extracting keyframes: %s",
                task.task_id, exc,
            )
            try:
                keyframe_paths = await extract_keyframes(video_path, task.task_id)
                logger.info("Extracted %d keyframes for task %s",
                           len(keyframe_paths), task.task_id)
            except Exception as kf_exc:
                logger.error("Keyframe extraction failed for task %s: %s",
                           task.task_id, kf_exc)
                await description_queue.put(None)
                return False

        # Step 4: Generate description
        try:
            description = await provider.generate_description(video_ref, keyframe_paths)
            logger.info("Generated description for task %s", task.task_id)
            await description_queue.put(DescriptionItem(task=task, description=description))
            return True
        except (GenerationError, Exception) as exc:
            logger.error("Description failed for task %s: %s", task.task_id, exc)
            await description_queue.put(None)
            return False

    finally:
        # Always return provider to pool
        await description_pool.put(provider)


# ---------- Stage 2: Caption Consumers ----------

async def consume_captions(
    description_queue: asyncio.Queue,
    caption_pool: asyncio.Queue,
    results: list[TaskResult],
    results_lock: asyncio.Lock,
    total_tasks: int,
    tasks_processed: asyncio.Event,
) -> None:
    """Consumer worker: pop descriptions, generate captions, collect results.

    Runs in a loop until all tasks are processed.

    Args:
        description_queue: Queue of DescriptionItem (or None for failures).
        caption_pool: Queue of (CaptionProvider, cooldown_seconds) tuples.
        results: Shared list to append TaskResults to.
        results_lock: Lock for safe append to results list.
        total_tasks: Total number of tasks expected.
        tasks_processed: Event set when all tasks have been processed.
    """
    processed_count = 0

    while True:
        # Check if all tasks are done
        if tasks_processed.is_set() and description_queue.empty():
            break

        try:
            item = await asyncio.wait_for(description_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            if tasks_processed.is_set():
                break
            continue

        if item is None:
            # Failed task — no description available, skip
            description_queue.task_done()
            continue

        # Get a caption model from the pool
        caption_provider, cooldown = await caption_pool.get()

        try:
            captions: dict[str, str] = {}
            for style in item.task.styles:
                try:
                    caption = await caption_provider.generate_caption(
                        item.description, style
                    )
                    caption = sanitize_caption(caption)
                    captions[style] = caption
                    logger.info("Generated %s caption for task %s",
                              style, item.task.task_id)
                except (GenerationError, Exception) as exc:
                    logger.warning(
                        "Caption failed for task %s style %s: %s",
                        item.task.task_id, style, exc,
                    )
                    captions[style] = FALLBACK_CAPTION

            result = TaskResult(task_id=item.task.task_id, captions=captions)
            async with results_lock:
                results.append(result)

        finally:
            # Return model to pool after cooldown
            async def _return_after_cooldown(provider, cd, pool):
                await asyncio.sleep(cd)
                await pool.put((provider, cd))

            asyncio.create_task(
                _return_after_cooldown(caption_provider, cooldown, caption_pool)
            )

        description_queue.task_done()


# ---------- Cleanup ----------

def _cleanup_temp_files() -> None:
    """Remove temporary video and keyframe files from /tmp/."""
    patterns = ["/tmp/*.mp4", "/tmp/*_frame_*.jpg"]
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
            except OSError:
                pass
    logger.info("Cleaned up temporary files")


# ---------- Main Orchestrator ----------

async def run_agent() -> None:
    """Main orchestrator: producer-consumer pipeline with model pools.

    Flow:
    1. Load model config and build provider pools
    2. Parse tasks
    3. Launch Stage 1 producers (descriptions) concurrently
    4. Launch Stage 2 consumers (captions) concurrently
    5. Wait for all work to complete
    6. Write results and cleanup
    """
    # Load model config
    try:
        config = load_model_config()
        logger.info("Loaded model configuration")
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load model config: %s", exc)
        sys.exit(1)

    # Build description provider pool
    try:
        desc_providers = build_description_providers(config)
        logger.info("Initialized %d description providers", len(desc_providers))
    except ValueError as exc:
        logger.error("Failed to init description providers: %s", exc)
        sys.exit(1)

    # Build caption provider pool
    try:
        caption_providers = build_caption_providers(config)
        logger.info("Initialized %d caption providers", len(caption_providers))
    except ValueError as exc:
        logger.error("Failed to init caption providers: %s", exc)
        sys.exit(1)

    # Parse tasks
    try:
        tasks = parse_tasks(INPUT_PATH)
        logger.info("Parsed %d tasks from %s", len(tasks), INPUT_PATH)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to parse tasks: %s", exc)
        sys.exit(1)

    if not tasks:
        write_results([], OUTPUT_PATH)
        logger.info("No tasks to process, wrote empty results")
        return

    # Build asyncio queues
    description_pool: asyncio.Queue = asyncio.Queue()
    for provider in desc_providers:
        await description_pool.put(provider)

    caption_pool: asyncio.Queue = asyncio.Queue()
    for provider, cooldown in caption_providers:
        await caption_pool.put((provider, cooldown))

    description_queue: asyncio.Queue = asyncio.Queue()

    # Shared state
    results: list[TaskResult] = []
    results_lock = asyncio.Lock()
    tasks_processed = asyncio.Event()
    start_time = time.monotonic()

    # Launch Stage 1: Description producers
    producer_tasks = [
        asyncio.create_task(
            produce_description(task, description_pool, description_queue, start_time)
        )
        for task in tasks
    ]

    # Launch Stage 2: Caption consumers (one per caption model)
    num_consumers = len(caption_providers)
    consumer_tasks = [
        asyncio.create_task(
            consume_captions(
                description_queue, caption_pool, results,
                results_lock, len(tasks), tasks_processed,
            )
        )
        for _ in range(num_consumers)
    ]

    # Wait for all producers to finish
    await asyncio.gather(*producer_tasks, return_exceptions=True)
    tasks_processed.set()
    logger.info("All description producers completed")

    # Wait for consumers to drain the queue
    await asyncio.gather(*consumer_tasks, return_exceptions=True)
    logger.info("All caption consumers completed")

    # Handle tasks that failed completely (no result entry)
    completed_task_ids = {r.task_id for r in results}
    for task in tasks:
        if task.task_id not in completed_task_ids:
            results.append(TaskResult(
                task_id=task.task_id,
                captions={style: FALLBACK_CAPTION for style in task.styles},
            ))

    # Write results
    write_results(results, OUTPUT_PATH)
    logger.info("Wrote %d results to %s", len(results), OUTPUT_PATH)

    # Cleanup
    _cleanup_temp_files()

    elapsed = time.monotonic() - start_time
    logger.info("Pipeline completed in %.1f seconds", elapsed)


if __name__ == "__main__":
    asyncio.run(run_agent())
