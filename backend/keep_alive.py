from upstash_redis import Redis
import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def trigger_activity():
    redis_url = os.getenv("UPSTASH_REDIS_REST_UR")
    if not redis_url:
        logger.error("Redis URL not found in environment variable")
        sys.exit(1)

    try:
        redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))
        redis.set("synq_system_status", "active")
        logger.info("Successfully pinged Upstash Redis for Synq")
    except Exception as e:
        logger.error(f"Error connecting to Upstash Redis {e}")
        sys.exit(1)

if __name__ == "__main__":
    trigger_activity()
        
