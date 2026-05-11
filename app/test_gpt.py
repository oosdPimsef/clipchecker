import os
import requests
import httpx
import msvcrt  # для работы с клавишами в Windows
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем API ключ из .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

proxy=os.getenv("OPENAI_PROXY_URL")
proxies = {
    'http': proxy,
    'https': proxy
}


if not api_key:
    raise ValueError("API ключ не найден в .env файле.")

print("=== Проверка IP и геолокации ===")
try:
    ip = requests.get("https://ifconfig.io/ip", proxies=proxies, timeout=5).text.strip()
    country = requests.get("https://ipinfo.io/country",proxies=proxies, timeout=5).text.strip()
    print(f"Ваш текущий IP: {ip}")
    print(f"Страна по данным GeoIP: {country}")
except Exception as e:
    print(f"Ошибка при определении IP: {e}")

print("\n=== Проверка запроса к OpenAI API ===")
client = OpenAI(api_key=api_key) if proxy is None or proxy == "" else OpenAI(http_client=httpx.Client(proxy=proxy), api_key=api_key)


try:
    resp = client.models.list()
    print("✅ API доступен. Список моделей получен.")
    print(f"Всего моделей: {len(resp.data)}")
except Exception as e:
    print(f"❌ Ошибка при обращении к API: {e}")

# Ждём нажатия пробела для выхода
print("\nНажмите пробел, чтобы закрыть окно...")
while True:
    if msvcrt.kbhit() and msvcrt.getch() == b' ':
        break