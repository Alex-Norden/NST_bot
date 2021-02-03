import os


API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")

# webhook settings
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH")
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# webserver settings
WEBAPP_HOST = os.getenv("WEBAPP_HOST")
WEBAPP_PORT = int(os.getenv("PORT"))

ADMIN_ID = int(os.getenv("ADMIN_ID"))

TARGET_SIZE1 = int(os.getenv("TARGET_SIZE1"))
TARGET_SIZE2 = int(os.getenv("TARGET_SIZE2"))
NUM_STEPS = int(os.getenv("NUM_STEPS", 500))