from helpers.config import get_settings
import uvicorn

settings = get_settings()


if __name__ == "__main__":
    host = settings.APP_HOST
    port = settings.APP_PORT
    workers = settings.WORKERS
    uvicorn.run(
        "app:app", host=host, port=port, reload=False, loop="asyncio", workers=workers
    )
