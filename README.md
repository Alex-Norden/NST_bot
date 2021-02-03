# NST_bot
Neural Style Transfer Bot For Telegram

Telegram bot для переноса стиля с одного изображения на другое.
Бот использует два алгоритма, которые описаны в статьях:
- "A Neural Algorithm of Artistic Style" (http://arxiv.org/abs/1508.06576)
- "Multi-style Generative Network for Real-time Transfer" (https://arxiv.org/abs/1703.06953)

Для деплоя использовался сервис Heroku ![Deploy](https://www.herokucdn.com/deploy/button.svg)

Для запуска бота необходимо задать переменные окружения:
- TELEGRAM_API_TOKEN - токен бота, полученный от [@BotFather](https://t.me/BotFather)
- WEBHOOK_HOST, WEBHOOK_PATH, которые образуют WEBHOOK_URL, на который необходимо установить вебхук
- WEBAPP_HOST, WEBAPP_PORT - хост и порт веб-приложения соответственно

Также дополнительно используются:
- ADMIN_ID - ID Telegram аккаунта для информирования о том, что бот перешёл в неактивное состояние
- TARGET_SIZE1, TARGET_SIZE2 - размеры изображений для первого и второго алгоритма соответственно, используются для скалирования с сохранением соотношения сторон
- NUM_STEPS - максимальное количество итераций для первого алгоритма

Веса моделей в файлах:
- [Gatys.model](resources/Gatys.model)
- [21styles.model](resources/21styles.model)

Для написания бота использовалась библиотека aiogram.
Вместо polling использовался webhook, т.к. это более эффективно.

Бот доступен в Telegram: [@ArtPicaBot](https://t.me/ArtPicaBot)
Добавлена краткая справка и подсказки.


Дополнительные ресурсы:

- https://github.com/pytorch/examples
- https://github.com/pytorch/tutorials/blob/master/advanced_source/neural_style_tutorial.py
- https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer
- https://core.telegram.org/
- https://docs.aiogram.dev/en/latest/
