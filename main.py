
import os
import json
import urllib.request
import urllib.parse

def send_telegram_message(message):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message
    }).encode()

    request = urllib.request.Request(url, data=data)

    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.loads(response.read().decode())

    if not result.get("ok"):
        raise RuntimeError("Telegram could not send the message.")

    print("Telegram message sent successfully.")

if __name__ == "__main__":
    send_telegram_message(
        "✅ Stock Alert Bot is connected!\n\n"
        "Telegram setup is working. Trading signals are not active yet."
    )
