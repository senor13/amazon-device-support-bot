"""
Seed MongoDB Atlas with the Kindle document tree fixture.

Usage:
    MONGODB_ATLAS_URI="mongodb+srv://..." python3 -m prep.seed.seed_atlas
"""

import asyncio
import json
import os
from pathlib import Path

import motor.motor_asyncio

FIXTURE = Path(__file__).parent / "document_trees.json"


async def main() -> None:
    uri = os.environ.get("MONGODB_ATLAS_URI")
    if not uri:
        raise SystemExit("Set MONGODB_ATLAS_URI env var before running this script.")

    client = motor.motor_asyncio.AsyncIOMotorClient(uri)
    db = client["support_bot"]

    data = [
        json.loads(line)
        for line in FIXTURE.read_text().splitlines()
        if line.strip()
    ]

    await db.document_trees.delete_many({})
    result = await db.document_trees.insert_many(data)
    print(f"Seeded {len(result.inserted_ids)} documents into Atlas.")
    client.close()


if __name__ == "__main__":
    asyncio.run(main())
