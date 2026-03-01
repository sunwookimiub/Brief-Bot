import asyncio, traceback, sys
sys.path.insert(0, ".")
from app.config import Settings
from app.ingestion.pipeline import IngestionPipeline

async def main():
    settings = Settings()
    print(f"Bucket: {settings.gcs_bucket}")
    pipeline = IngestionPipeline(settings)
    try:
        result = await pipeline.run(gcs_prefix="manuals/test/", doc_version="v1")
        print("Result:", result)
    except Exception as e:
        traceback.print_exc()

asyncio.run(main())
