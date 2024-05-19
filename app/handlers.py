import json, os, time, re

from aiogram import Bot, Router, types, F
from aiogram.filters.command import Command, CommandObject
from aiogram.types import CallbackQuery, FSInputFile, Message
from typing import Any, Callable, Dict, Awaitable


rt = Router()

from dotenv import load_dotenv

from app.soundcloud_scraper import mus_search
from app.data_class import mus_class

load_dotenv()


max_length = 4096

with open('data/phrases.json', 'r', encoding='utf-8') as file:
    fras = json.load(file)

bot = Bot(os.getenv('TOKEN'))
delay_message = int(os.getenv('DELAY'))

user_last_message_time = {}



@rt.message.outer_middleware()
async def Antiflood(
    handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
    message: Message,
    data: Dict[str, Any]):

    if message.from_user.id not in user_last_message_time:
        user_last_message_time[message.from_user.id] = 0

    if (int(time.time()) - user_last_message_time[message.from_user.id]) > delay_message:
        user_last_message_time[message.from_user.id] = int(time.time())
        return await handler(message, data)


@rt.message(Command("start"))
async def cmd_start(message: types.Message, command: CommandObject):
    await bot.unpin_all_chat_messages(chat_id = message.from_user.id)
    pinmes = await message.answer(fras["start"])
    return await bot.pin_chat_message(message.from_user.id, pinmes.message_id, False)


@rt.message(F.text)
async def mess_handler(message: types.Message):
    await bot.send_chat_action(message.from_user.id, "typing")
    url = str(message.text)
    soundcloud_pattern = r'^https?:\/\/(?:www\.)?soundcloud\.com\/.*$'
    if re.match(soundcloud_pattern, url):
        playlist = await mus_search(url)
        analys = await mus_class(playlist)
        await message.answer(analys)
    else:
        return await message.answer(fras["errorurl"])
