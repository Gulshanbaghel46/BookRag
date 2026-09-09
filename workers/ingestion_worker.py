"""Background worker: PDF/text extraction, chunking, embeddings, pgvector storage."""
import asyncio
import json
import logging

import aio_pika

from app.config import get_settings
from app.services.document_service import document_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


async def process_message(message: aio_pika.IncomingMessage):
    async with message.process():
        data = json.loads(message.body.decode())
        document_id = data["document_id"]
        file_path = data["file_path"]
        file_name = data["file_name"]
        metadata = data.get("metadata") or {}
        logger.info("Processing document %s: %s", document_id, file_name)
        await document_service.process_document(
            document_id=document_id,
            file_path=file_path,
            file_name=file_name,
            metadata=metadata,
        )
        logger.info("Successfully processed document %s", document_id)


async def main():
    logger.info("Starting document ingestion worker...")
    await document_service.initialize()
    connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
    logger.info("Worker connected to RabbitMQ")

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue = await channel.declare_queue("document_processing", durable=True)
        logger.info("Worker ready, waiting for documents...")
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                try:
                    await process_message(message)
                except Exception as e:
                    logger.error("Failed to process message: %s", e)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
