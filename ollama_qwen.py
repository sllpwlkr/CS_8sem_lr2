import requests
import json
from datetime import datetime

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"


def ask_llm(prompt: str) -> str:

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        # Отправляем POST-запрос к API Ollama
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()

        # Парсим JSON-ответ и извлекаем поле 'response'
        answer = response.json().get("response", "[Ошибка: Поле 'response' не найдено]")
        return answer.strip()

    except requests.exceptions.Timeout:
        return "Ошибка: Таймаут при ожидании ответа от Ollama."
    except requests.exceptions.ConnectionError:
        return "Ошибка: Не удалось подключиться к серверу Ollama. Убедитесь, что он запущен."
    except requests.exceptions.RequestException as e:
        return f"Ошибка при выполнении запроса: {e}"
    except json.JSONDecodeError:
        return "Ошибка: Не удалось разобрать ответ от сервера."


def save_results_to_md(results, filename="inference_report.md"):

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Отчёт инференса: {MODEL_NAME}\n\n")
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Результаты запросов\n\n")
        f.write("| № | Запрос к LLM | Ответ LLM |\n")
        f.write("|---|--------------|------------|\n")

        for num, question, answer in results:
            answer_md = answer.replace('\n', '<br>')
            question_escaped = question.replace('|', '\\|')
            answer_escaped = answer_md.replace('|', '\\|')
            f.write(f"| {num} | {question_escaped} | {answer_escaped} |\n")

        f.write("\n---\n")
        f.write(f"*Всего обработано запросов: {len(results)}*\n")

    print(f"Результаты сохранены в файл: {filename}")


def main():

    questions = [
        "Реши уравнение 17 - 12 * 2 - 2/4",
        "Сколько лет живут черепахи?",
        "Сколько будет семь умножить на восемь",
        "Что такое алгоритм?",
        "Назови столицу России",
        "Сколько дней в неделе?",
        "Кто такие BTS?",
        "Как сказать здравствуйте на французском?",
        "Как купить билеты на концерт?",
        "Что значит «билд» в играх?",

    ]
    print("=" * 60)
    print("Запуск тестирования модели Qwen2.5:0.5B")
    print("=" * 60)

    results = []  # для сохранения (номер, вопрос, ответ)

    for i, question in enumerate(questions, 1):
        print(f"\n--- Запрос №{i}: {question} ---")
        answer = ask_llm(question)
        print(f"--- Ответ:\n{answer}\n")
        results.append((i, question, answer))

    print("=" * 60)
    print("Тестирование завершено.")
    print("=" * 60)

    save_results_to_md(results)


if __name__ == "__main__":
    main()