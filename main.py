import asyncio
import logging
import os

from aiogram import Bot, Dispatcher

from dotenv import load_dotenv
from app.handlers import rt


load_dotenv()

bot = Bot(os.getenv('TOKEN'))
dp = Dispatcher()

logging.basicConfig(level=logging.INFO)

dp.include_router(rt)


async def main():

    await bot.delete_webhook(drop_pending_updates=True)

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())