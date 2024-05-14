# RAG system



## Локальный запуск проекта

Python 3.11

```bash
# развернуть и активировать виртуальное окружение
python -m venv venv
source venv/bin/activate

# установить зависимости
pip install -r requirements.txt

# запустить api приложение
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Запуск проекта в докер контейнере

```bash
docker-compose build
docker-compose up
```

## Применение

На эндпоинт `http://localhost:8000/generate` отправить POST запрос

```json
{
  "query": "Какой рейтинг при открытии ПВЗ?"
}
```

Запрос можно отправить через Postman, либо воспользовавшись Swagger'ом по адресу `http://localhost:8000/docs`

Перечень вопросов в колонке questions в [QA_pairs.xlsx](data/QA_pairs.xlsx)
