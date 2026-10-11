
import os
import json
import urllib.request
import urllib.parse
import urllib.error


def send_telegram_message(message):
    token = os.environ["TELEGRAM_BOT_TOKEN"].strip()
    chat_id = os.environ["TELEGRAM_CHAT_ID"].strip()

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is empty.")

    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is empty.")

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message
    }).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))

        if not result.get("ok"):
            raise RuntimeError(
                "Telegram API returned an unsuccessful response."
            )

        print("SUCCESS: Telegram message sent.")

    except urllib.error.HTTPError as error:
        # Do not print the request URL because it contains your bot token.
        print(f"Telegram API returned HTTP {error.code}.")
        raise RuntimeError(
            "Telegram rejected the request. Check the bot token and chat ID."
        ) from None


if __name__ == "__main__":
    send_telegram_message(
        "Stock Alert Bot is connected!\n\n"
        "Telegram setup is working. Trading signals are not active yet."
    )
