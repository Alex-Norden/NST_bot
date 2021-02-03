"""
CPU style transfer bot
"""
import os
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import Message, ContentTypes, InputFile
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils.executor import start_webhook

import config
from utils import DEFINED_STYLES, STYLE_CMDS, PATH
from task_thread import TaskThread


# configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


# initialize bot
bot = Bot(token=config.API_TOKEN)
bot.consumer = None
dp = Dispatcher(bot, storage=MemoryStorage())
queue = asyncio.Queue()
task_thread = TaskThread()


# states
class Form(StatesGroup):
	load_content = State()
	load_style = State()
	set_style = State()
	confirm = State()
	transfer = State()


def generator_id():
	c = 1
	while True:
		yield c
		c += 1

gen_id = generator_id()


async def to_confirm(_id):
	await bot.send_message(_id, "Для продолжения /continue или вернуться /cancel")
	await Form.confirm.set()

async def check_task(state):
	data = await state.get_data()
	task_id = data.get("task_id")
	if task_id:
		position = task_id - task_thread.last_task_id
		return f"Подождите, ещё не закончил, позиция в очереди: {position}"


# =================== BEGIN HANDLERS ==========================
@dp.message_handler(state="*", commands=["start", "help"])
async def process_welcome(msg: Message):
	await msg.answer("Бот для переноса стиля с одной картинки на другую.\nНеобходимо выбрать алгоритм и следовать подсказкам\n/first_transfer - A Neural Algorithm of Artistic Style\n/second_transfer - Multi-style Generative Network for Real-time Transfer\n/cancel - вернуться в начало")

@dp.message_handler(state="*", commands=["first_transfer", "second_transfer"])
async def process_transfer_mode(msg: Message, state: FSMContext):
	resp = await check_task(state)
	if resp:
		return await msg.answer(resp)

	await state.update_data(is_first=bool(msg.text.lower() == "/first_transfer"))

	await msg.answer("Отправьте мне картинку контента")
	await Form.load_content.set()

@dp.message_handler(state=Form.load_content, content_types=["photo"])
async def process_content_img(msg: Message, state: FSMContext):
	data = await state.get_data()

	file_id = msg.photo[-1].file_id
	await state.update_data(content_id=file_id)

	if data.get("is_first"):
		await msg.answer("Отправьте мне картинку стиля")
		await Form.load_style.set()
	else:
		await msg.answer("Выберите стиль:\n{}".format("\n".join(STYLE_CMDS.keys())))
		await Form.set_style.set()

@dp.message_handler(state=Form.load_style, content_types=["photo"])
async def process_style_img(msg: Message, state: FSMContext):
	file_id = msg.photo[-1].file_id
	await state.update_data(style_id=file_id)
	await to_confirm(msg.chat.id)

@dp.message_handler(state=Form.set_style, commands=DEFINED_STYLES)
async def process_style(msg: Message, state: FSMContext):
	style_num = STYLE_CMDS[msg.text.lower()]
	style_path = os.path.join(PATH.DEFINED_STYLES_DIR, f"{DEFINED_STYLES[style_num]}.jpg")
	await msg.answer_photo(InputFile(style_path), caption="Выбранный стиль") #show style
	await state.update_data(style_num=style_num)
	await to_confirm(msg.chat.id)

@dp.message_handler(state=Form.confirm, commands=["continue"])
async def process_continue(msg: Message, state: FSMContext):
	data = await state.get_data()
	await state.update_data(task_id=next(gen_id))
	await queue.put(msg.chat.id)

	await msg.answer("Подождите несколько {}".format("минут" if data.get("is_first") else "секунд"))
	await Form.transfer.set()

# you can use state "*" if you need to handle all states
@dp.message_handler(state="*", commands="cancel")
@dp.message_handler(Text(equals="cancel", ignore_case=True), state="*")
async def cancel_handler(msg: Message, state: FSMContext):
	current_state = await state.get_state()
	if current_state is None:
		return await msg.answer("Пока ничего не выбрано")

	# check processed
	data = await state.get_data()
	processed = data.get("processed")
	if processed:
		return await msg.answer("Вы не можете отменить, идёт обработка")

	LOG.info(f"Cancelling state: {current_state}")
	# cancel state and inform user about it
	await state.finish()
	await msg.answer("Отменено")

@dp.message_handler(state=Form.transfer, content_types=ContentTypes.ANY)
async def process_transfer(msg: Message, state: FSMContext):
	resp = await check_task(state)
	if resp:
		await msg.answer(resp)
	else:
		LOG.error(f"Failed process_transfer by id: {msg.chat.id}")
		await state.finish()
		await msg.answer("Что-то пошло не так, попробуйте заново")

@dp.message_handler(state="*", content_types=ContentTypes.ANY)
async def default_handler(msg: Message):
	await msg.answer("Не понимаю, посмотри /help")
# =================== END HANDLERS ==========================


async def download_img(file_id, save_path):
	file = await bot.get_file(file_id)
	await bot.download_file(file.file_path, save_path)

async def do_task(_id, state):
	data = await state.get_data()
	if not data:
		LOG.error(f"No such data by id: {_id}")
		return

	task_id = data.get("task_id")
	if task_id is None:
		LOG.error(f"No task_id by id: {_id}")
		return

	content_id = data.get("content_id")
	if content_id is None:
		LOG.error(f"No content_id by id: {_id}")
		return
	await download_img(content_id, PATH.CONTENT_PATH)

	is_first = data.get("is_first")
	if is_first is None:
		LOG.error(f"No mode by id: {_id}")
		return

	if is_first:
		style_id = data.get("style_id")
		if style_id is None:
			LOG.error(f"No style_id by id: {_id}")
			return
		await download_img(style_id, PATH.STYLE_PATH)
	else:
		style_num = data.get("style_num")
		if style_num is None:
			LOG.error(f"No style_num by id: {_id}")
			return
		task_thread.style_num = style_num

	await state.update_data(processed=True)

	task_thread.is_first = is_first
	task_thread.has_task = True

	while task_thread.has_task:
		await asyncio.sleep(2)

	task_thread.last_task_id = task_id

	return task_thread.is_success


async def consumer(queue):
	while True:
		if not task_thread.has_task:
			_id = await queue.get()
			LOG.info(f"received {_id}")

			try:
				state = dp.current_state(user=_id)

				try:
					is_success = await do_task(_id, state)
				except Exception as e:
					LOG.critical(f"task exception: {e}")
					is_success = False

				if is_success:
					await bot.send_photo(_id, InputFile(PATH.RESULT_PATH), caption="Готово!")
				else:
					await bot.send_message(_id, "Что-то пошло не так, попробуйте заново")

				await state.finish()
			except Exception as e:
				LOG.critical(f"consumer exception: {e}")

			queue.task_done()
		else:
			await asyncio.sleep(2)


async def on_startup(dp: Dispatcher):
	LOG.warning("on_startup")
	await bot.delete_webhook()
	await bot.set_webhook(config.WEBHOOK_URL)
	# insert code here to run it after start
	if not os.path.isdir(PATH.USERDATA_DIR):
		try:
			os.makedirs(PATH.USERDATA_DIR)
		except OSError as e:
			raise OSError(f"Failed make directory: {e}")
	else:
		LOG.warning(f"directory is already exists: {PATH.USERDATA_DIR}")

	task_thread.start()
	bot.consumer = asyncio.create_task(consumer(queue))

async def on_shutdown(dp: Dispatcher):
	LOG.warning("on_shutdown")

	await bot.send_message(config.ADMIN_ID, "Shutting down..")

	# close all
	await dp.storage.close()
	await dp.storage.wait_closed()

	if bot.consumer:
		bot.consumer.cancel()
		LOG.info("consumer canceled")

	if task_thread.is_alive():
		task_thread.running = False
		task_thread.join(1000)
		LOG.info("task_thread stoped")

	LOG.warning("bye!")


if __name__ == "__main__":
	# executor.start_polling(dp, skip_updates=True, on_shutdown=on_shutdown)
	start_webhook(
		dispatcher=dp,
		webhook_path=config.WEBHOOK_PATH,
		on_startup=on_startup,
		on_shutdown=on_shutdown,
		skip_updates=True,
		host=config.WEBAPP_HOST,
		port=config.WEBAPP_PORT,
	)