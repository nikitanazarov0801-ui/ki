import os
import re
import json
import time
import sqlite3
import hashlib
import logging
from logging.handlers import TimedRotatingFileHandler
from urllib.parse import urlparse
from io import BytesIO
from math import ceil

import requests
import feedparser
from bs4 import BeautifulSoup
from PIL import Image

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


# -------------------- LOGGING --------------------
def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("pro_kosmos")
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    log_path = os.getenv("LOG_PATH", "pro_kosmos.log")
    fh = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        interval=1,
        backupCount=int(os.getenv("LOG_BACKUP_DAYS", "14")),
        encoding="utf-8",
        utc=False
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(sh)
        logger.addHandler(fh)

    return logger


log = setup_logging()


# -------------------- CONFIG --------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
TARGET_CHANNEL_ID = os.getenv("TARGET_CHANNEL_ID")  # -100...
YC_API_KEY = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")

TZ = os.getenv("TZ", "Europe/Moscow")
RUN_NOW = os.getenv("RUN_NOW", "0") == "1"

RSS_TIMEOUT = float(os.getenv("RSS_TIMEOUT", "12"))
PAGE_TIMEOUT = float(os.getenv("PAGE_TIMEOUT", "12"))
YC_TIMEOUT = float(os.getenv("YC_TIMEOUT", "60"))

# Separate image download timeouts/retries
IMG_CONNECT_TIMEOUT = float(os.getenv("IMG_CONNECT_TIMEOUT", "5"))
IMG_READ_TIMEOUT = float(os.getenv("IMG_READ_TIMEOUT", "60"))
IMG_RETRIES = int(os.getenv("IMG_RETRIES", "3"))
IMG_RETRY_SLEEP = float(os.getenv("IMG_RETRY_SLEEP", "2.0"))

RSS_UA = os.getenv("RSS_UA", "PROKosmosBot/2.4")

LIMIT_PER_FEED = int(os.getenv("LIMIT_PER_FEED", "8"))
CANDIDATE_PACK_LIMIT = int(os.getenv("PACK_LIMIT", "30"))

# Caption limit for photos: 0–1024 after entities parsing
CAPTION_LIMIT = int(os.getenv("CAPTION_LIMIT", "900"))

# Telegram-ish safe targets for sendPhoto
TG_PHOTO_TARGET_BYTES = int(os.getenv("TG_PHOTO_TARGET_BYTES", str(9 * 1024 * 1024)))
MAX_ORIGINAL_BYTES = int(os.getenv("MAX_ORIGINAL_BYTES", str(25 * 1024 * 1024)))
MAX_DIM = int(os.getenv("MAX_DIM", "1600"))

# "Don't lose slot": how many candidates to try per run
POST_TRIES = int(os.getenv("POST_TRIES", "3"))

DB_PATH = os.getenv("DB_PATH", "seen.db")

RSS = [
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "https://spaceflightnow.com/feed",
    "https://space.com/feeds.xml",
    "https://ria.ru/export/rss2/index.xml",
    "https://nplus1.ru/rss",
    "https://naked-science.ru/allrss",
]


# -------------------- UTILS --------------------
def strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sha1hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc or "source"
    except Exception:
        return "source"


# -------------------- STORAGE --------------------
def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("CREATE TABLE IF NOT EXISTS seen (k TEXT PRIMARY KEY, ts INTEGER)")
    db.commit()
    return db

DB = init_db()

def seen_before(key: str) -> bool:
    cur = DB.execute("SELECT 1 FROM seen WHERE k=?", (key,))
    return cur.fetchone() is not None

def mark_seen(key: str):
    DB.execute("INSERT OR IGNORE INTO seen(k, ts) VALUES(?, ?)", (key, int(time.time())))
    DB.commit()


# -------------------- TELEGRAM --------------------
def tg_send_message(text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    r = requests.post(
        url,
        json={
            "chat_id": TARGET_CHANNEL_ID,
            "text": text,
            "disable_web_page_preview": False,
        },
        timeout=30,
    )
    if not r.ok:
        log.error("Telegram sendMessage HTTP %s: %s", r.status_code, r.text[:3000])
    r.raise_for_status()

def tg_send_photo_bytes(photo_bytes: bytes, caption: str, filename: str = "photo.jpg"):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    data = {
        "chat_id": TARGET_CHANNEL_ID,
        "caption": caption,
        "disable_notification": False,
    }
    files = {"photo": (filename, photo_bytes)}
    r = requests.post(url, data=data, files=files, timeout=120)
    if not r.ok:
        log.error("Telegram sendPhoto HTTP %s: %s", r.status_code, r.text[:3000])
    r.raise_for_status()


# -------------------- RSS --------------------
def fetch_feed(feed_url: str):
    r = requests.get(
        feed_url,
        headers={"User-Agent": RSS_UA},
        timeout=(5, RSS_TIMEOUT),
    )
    r.raise_for_status()
    return feedparser.parse(BytesIO(r.content))


# -------------------- IMAGE FINDING (NO VIDEO) --------------------
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif")

def _is_image_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower().split("?")[0]
    return u.endswith(IMAGE_EXT)

def extract_image_url(entry, link: str) -> str | None:
    """
    Returns ONLY an image URL (mp4 ignored).
    1) media:content / media:thumbnail (image/* or *.jpg/png/..)
    2) enclosures (image/*)
    3) <img src> in summary
    4) og:image on page
    """
    try:
        for key in ("media_content", "media_thumbnail"):
            mc = entry.get(key)
            if isinstance(mc, list):
                for item in mc:
                    url = item.get("url") or item.get("href")
                    mime = (item.get("type") or "").lower()
                    if mime.startswith("image/") or _is_image_url(url):
                        return url

        enclosures = entry.get("enclosures") or []
        for e in enclosures:
            href = e.get("href")
            typ = (e.get("type") or "").lower()
            if href and (typ.startswith("image/") or _is_image_url(href)):
                return href

        summary = entry.get("summary", "") or ""
        m = re.search(r'<img[^>]+src="([^"]+)"', summary)
        if m:
            u = m.group(1)
            if _is_image_url(u):
                return u

    except Exception:
        pass

    if link:
        try:
            r = requests.get(
                link,
                headers={"User-Agent": RSS_UA},
                timeout=(5, PAGE_TIMEOUT),
            )
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            tag = soup.find("meta", attrs={"property": "og:image"})
            if tag and tag.get("content"):
                og = tag["content"].strip()
                if _is_image_url(og):
                    return og
        except Exception:
            log.info("No og:image (or page fetch failed): %s", link)

    return None


# -------------------- IMAGE VARIANTS (NASA) --------------------
def expand_image_variants(url: str) -> list[str]:
    """
    NASA images-assets часто отдаёт *~large.jpg, который может качаться долго.
    Пробуем варианты поменьше.
    """
    if not url:
        return []

    variants = [url]

    if "images-assets.nasa.gov" in url and "~large" in url:
        variants = [
            url.replace("~large", "~medium"),
            url.replace("~large", "~small"),
            url,
        ]

    seen = set()
    out = []
    for u in variants:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


# -------------------- IMAGE DOWNLOAD + NORMALIZE FOR TELEGRAM --------------------
def _download_bytes_limited(url: str, max_bytes: int) -> bytes | None:
    # timeout tuple: (connect, read)
    r = requests.get(
        url,
        headers={"User-Agent": RSS_UA},
        stream=True,
        timeout=(IMG_CONNECT_TIMEOUT, IMG_READ_TIMEOUT),
    )
    r.raise_for_status()

    buf = bytearray()
    for chunk in r.iter_content(chunk_size=128 * 1024):
        if not chunk:
            continue
        buf.extend(chunk)
        if len(buf) > max_bytes:
            return None
    return bytes(buf)

def _download_bytes_limited_retry(url: str, max_bytes: int) -> bytes | None:
    last_exc = None
    for attempt in range(1, IMG_RETRIES + 1):
        try:
            return _download_bytes_limited(url, max_bytes)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            log.warning("Image download attempt %d/%d failed: %s", attempt, IMG_RETRIES, repr(e))
            time.sleep(IMG_RETRY_SLEEP)

    if last_exc:
        raise last_exc
    return None

def _pad_to_ratio_20(im: Image.Image) -> Image.Image:
    w, h = im.size
    if w <= 0 or h <= 0:
        return im

    ratio = max(w / h, h / w)
    if ratio <= 20:
        return im

    # Pad with white background to avoid extreme ratio
    if w > h:
        new_h = ceil(w / 20)
        bg = Image.new("RGB", (w, new_h), (255, 255, 255))
        bg.paste(im, (0, (new_h - h) // 2))
        return bg
    else:
        new_w = ceil(h / 20)
        bg = Image.new("RGB", (new_w, h), (255, 255, 255))
        bg.paste(im, ((new_w - w) // 2, 0))
        return bg

def _resize_to_sum_10000(im: Image.Image) -> Image.Image:
    w, h = im.size
    if w <= 0 or h <= 0:
        return im

    s = w + h
    if s <= 10000:
        return im

    scale = 10000 / s
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return im.resize((new_w, new_h), Image.LANCZOS)

def download_and_prepare_photo(img_url: str) -> tuple[bytes | None, str]:
    """
    Always re-encode to a Telegram-safe JPEG:
      - ratio <= 20 (pad)
      - w+h <= 10000 (resize)
      - file size <= TG_PHOTO_TARGET_BYTES (quality loop)
    """
    filename = (img_url.split("?")[0].split("/")[-1] or "photo.jpg")
    if "." not in filename:
        filename += ".jpg"

    data = None
    chosen_url = None

    # Try variants first (for NASA)
    for u in expand_image_variants(img_url):
        try:
            data = _download_bytes_limited_retry(u, MAX_ORIGINAL_BYTES)
            if data is not None:
                chosen_url = u
                break
        except Exception:
            continue

    # Fallback direct (to get proper exception/logging if needed)
    if data is None:
        data = _download_bytes_limited_retry(img_url, MAX_ORIGINAL_BYTES)
        if data is None:
            return None, filename
        chosen_url = img_url

    try:
        im = Image.open(BytesIO(data))

        # Convert to RGB, handle alpha with white background
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")

        im = _pad_to_ratio_20(im)
        im = _resize_to_sum_10000(im)

        # extra safety
        im.thumbnail((MAX_DIM, MAX_DIM))

        last = None
        for q in (90, 85, 80, 75, 70, 65, 60, 55, 50):
            out = BytesIO()
            im.save(out, format="JPEG", quality=q, optimize=True)
            jpg = out.getvalue()
            last = jpg
            if len(jpg) <= TG_PHOTO_TARGET_BYTES:
                return jpg, "photo.jpg"

        if last:
            return last, "photo.jpg"

        return None, (chosen_url.split("?")[0].split("/")[-1] or filename)

    except Exception:
        return None, filename


# -------------------- YANDEXGPT --------------------
def _extract_yc_text(payload: dict) -> str:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected YandexGPT payload type: {type(payload)}")

    if "error" in payload:
        raise RuntimeError(f"YandexGPT error: {payload['error']}")

    container = payload
    for k in ("result", "response"):
        if isinstance(payload.get(k), dict):
            container = payload[k]
            break

    alts = container.get("alternatives")
    if not isinstance(alts, list) or not alts:
        raise RuntimeError(f"YandexGPT response has no alternatives. keys={list(container.keys())}")

    msg = alts[0].get("message") or {}
    text = msg.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"YandexGPT message has no text. message_keys={list(msg.keys())}")

    return text

def yc_completion(messages, temperature=0.2, max_tokens="900", json_schema=None, json_object=False):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {"Authorization": f"Api-Key {YC_API_KEY}"}

    body = {
        "modelUri": f"gpt://{YC_FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": float(temperature),
            "maxTokens": str(max_tokens),
        },
        "messages": messages,
    }

    if json_object:
        body["json_object"] = True
    if json_schema is not None:
        body["json_schema"] = {"schema": json_schema}

    r = requests.post(url, headers=headers, json=body, timeout=YC_TIMEOUT)
    if not r.ok:
        log.error("YandexGPT HTTP %s: %s", r.status_code, r.text[:3000])
    r.raise_for_status()

    payload = r.json()
    return _extract_yc_text(payload)


# -------------------- SELECTION/GENERATION --------------------
SPACE_KEYWORDS = [
    "космос", "космическ", "орбит", "спутник", "ракета", "носител",
    "луна", "марс", "венер", "юпитер", "сатурн", "астероид", "комет",
    "звезд", "галактик", "телескоп", "обсерватор", "мкс", "iss",
    "nasa", "esa", "spacex", "roscosmos", "роскосмос", "starship", "falcon",
]

def cheap_prefilter(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in SPACE_KEYWORDS)

def collect_candidates():
    candidates = []

    for feed_url in RSS:
        log.info("Fetching RSS: %s", feed_url)
        t0 = time.time()

        try:
            feed = fetch_feed(feed_url)
        except requests.exceptions.Timeout:
            log.warning("RSS timeout (%.1fs): %s", RSS_TIMEOUT, feed_url)
            continue
        except Exception:
            log.exception("RSS fetch/parse failed: %s", feed_url)
            continue

        entries = getattr(feed, "entries", []) or []
        log.info("Fetched RSS: %s in %.2fs entries=%d", feed_url, time.time() - t0, len(entries))

        for e in entries[:LIMIT_PER_FEED]:
            title = strip_html(getattr(e, "title", ""))
            summary_raw = getattr(e, "summary", "") or ""
            summary = strip_html(summary_raw)

            link = getattr(e, "link", "") or ""
            guid = getattr(e, "id", "") or link or f"{title}|{summary}|{feed_url}"
            key = sha1hex(guid)

            if seen_before(key):
                continue

            text = f"{title}\n{summary}\n{link}".strip()
            if not cheap_prefilter(text):
                continue

            candidates.append({
                "key": key,
                "title": title[:180],
                "summary": summary[:500],
                "link": link,
                "source": host_of(link) or host_of(feed_url),
                "entry": e,
            })

            if len(candidates) >= CANDIDATE_PACK_LIMIT:
                return candidates

    return candidates


PICK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "is_space": {"type": "boolean"},
        "pick": {"type": "integer", "minimum": -1},
        "reason": {"type": "string"}
    },
    "required": ["is_space", "pick", "reason"]
}

def pick_best(candidates):
    if not candidates:
        return None

    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"{i}) [{c['source']}] {c['title']}\n"
            f"Кратко: {c['summary']}\n"
            f"Ссылка: {c['link']}"
        )
    pack = "\n\n".join(lines)

    prompt = (
        "Выбери ОДНУ самую важную/интересную новость строго про космос/астрономию/космонавтику.\n"
        "Если подходящих нет — верни is_space=false и pick=-1.\n"
        "Ответь ТОЛЬКО валидным JSON по схеме.\n\n"
        f"Кандидаты:\n{pack}"
    )

    log.info("YandexGPT: picking best from %d candidates...", len(candidates))
    raw = yc_completion(
        messages=[
            {"role": "system", "text": "Ты редактор канала про космос. Строго JSON."},
            {"role": "user", "text": prompt},
        ],
        temperature=0.0,
        max_tokens="350",
        json_schema=PICK_SCHEMA
    )
    log.info("YandexGPT: picker done (%d chars)", len(raw))

    try:
        obj = json.loads(raw)
    except Exception:
        log.warning("Picker returned non-JSON: %r", raw[:2000])
        return None

    if obj.get("is_space") is False:
        log.info("Picker decided: is_space=false -> skip slot; reason=%s", obj.get("reason", "")[:200])
        return None

    pick = obj.get("pick", -1)
    if pick == -1:
        log.info("Picker decided: pick=-1 -> skip slot; reason=%s", obj.get("reason", "")[:200])
        return None

    if not isinstance(pick, int) or pick < 0 or pick >= len(candidates):
        log.warning(
            "Picker returned invalid pick=%r (candidates=%d) reason=%s",
            pick, len(candidates), obj.get("reason", "")[:200]
        )
        return None

    log.info("Picked index=%d reason=%s", pick, obj.get("reason", "")[:200])
    return candidates[pick]

def generate_post(c):
    src = (
        f"Заголовок: {c['title']}\n"
        f"Краткое описание: {c['summary']}\n"
        f"Источник: {c['link']}\n"
    )

    instr = (
        "Сгенерируй пост для Telegram-канала PRO Космос на русском.\n"
        "ФОРМАТ (строго):\n"
        "1) Заголовок (<= 80 знаков)\n"
        "2) 'Факты:' и ровно 2–3 буллета\n"
        "3) 'Почему важно:' (1 строка)\n"
        "4) 'Что дальше:' (1 строка)\n"
        "5) В конце строка 'Источник: <ссылка>'\n\n"
        "ОГРАНИЧЕНИЯ:\n"
        "- Не копируй дословно, перефразируй.\n"
        "- Не добавляй фактов, чисел и причин, которых нет во входных данных.\n"
        "- Если данных мало — так и напиши (без выдумок).\n"
    )

    log.info("YandexGPT: generating post...")
    text = yc_completion(
        messages=[
            {"role": "system", "text": "Ты редактор научпоп-канала. Пиши коротко и ясно."},
            {"role": "user", "text": instr + "\n\nВходные данные:\n" + src},
        ],
        temperature=0.35,
        max_tokens="900"
    ).strip()
    log.info("YandexGPT: post generated (%d chars)", len(text))
    return text


# -------------------- POSTING (ONLY WITH IMAGE) --------------------
def prepare_photo_for_candidate(c) -> tuple[bytes, str] | None:
    img_url = c.get("img_url") or extract_image_url(c.get("entry") or {}, c.get("link", ""))
    if not img_url:
        return None

    log.info("Image candidate: %s", img_url)

    try:
        photo_bytes, fname = download_and_prepare_photo(img_url)
    except Exception:
        log.exception("Skip: image download/prepare failed url=%s", img_url)
        return None

    if not photo_bytes:
        log.info("Skip: image unavailable/too large url=%s", img_url)
        return None

    return photo_bytes, fname

def post_prepared_to_telegram(photo_bytes: bytes, fname: str, post_text: str) -> bool:
    caption = post_text[:CAPTION_LIMIT]
    try:
        tg_send_photo_bytes(photo_bytes, caption, filename=fname)
    except Exception:
        log.exception("Skip: sendPhoto failed (treated as no-image)")
        return False

    if len(post_text) > CAPTION_LIMIT:
        tg_send_message(post_text[CAPTION_LIMIT:])

    return True


def run_job():
    started = time.time()
    log.info("Job started")

    try:
        candidates = collect_candidates()
        log.info("Collected candidates=%d (pack_limit=%d)", len(candidates), CANDIDATE_PACK_LIMIT)

        if not candidates:
            log.info("No candidates -> skip slot")
            return

        # 1) Keep only candidates with a real image URL (mp4 ignored by extract_image_url)
        candidates_img = []
        for c in candidates:
            img = extract_image_url(c.get("entry") or {}, c.get("link", ""))
            if img:
                c["img_url"] = img
                candidates_img.append(c)

        log.info("Candidates with images=%d", len(candidates_img))
        if not candidates_img:
            log.info("No candidates with images -> skip slot")
            return

        # 2) Pick best once (YandexGPT)
        best = pick_best(candidates_img)
        if not best:
            log.info("No best candidate -> skip slot")
            return

        # 3) Order: best first, then others
        ordered = [best] + [c for c in candidates_img if c["key"] != best["key"]]

        attempts = 0
        for c in ordered:
            if attempts >= POST_TRIES:
                break
            if seen_before(c["key"]):
                continue

            attempts += 1
            log.info(
                "Try %d/%d candidate source=%s link=%s",
                attempts, POST_TRIES, c.get("source"), c.get("link")
            )

            # (a) First: try to get a valid photo
            prepared = prepare_photo_for_candidate(c)
            if not prepared:
                # do NOT mark seen: CDN timeout can be temporary
                continue

            photo_bytes, fname = prepared

            # (b) Generate text only if photo is ready
            post_text = generate_post(c)

            # (c) Send photo; if sendPhoto fails, mark seen (likely permanent for this media)
            if post_prepared_to_telegram(photo_bytes, fname, post_text):
                mark_seen(c["key"])
                log.info("Posted OK source=%s link=%s", c.get("source"), c.get("link"))
                return
            else:
                mark_seen(c["key"])
                continue

        log.info("Skip slot: no candidates succeeded (attempted=%d)", attempts)

    except Exception:
        log.exception("Job failed with exception")
    finally:
        log.info("Job finished in %.2fs", time.time() - started)


# -------------------- SCHEDULER --------------------
def main():
    if not all([BOT_TOKEN, TARGET_CHANNEL_ID, YC_API_KEY, YC_FOLDER_ID]):
        raise RuntimeError("Заполни env: BOT_TOKEN, TARGET_CHANNEL_ID, YC_API_KEY, YC_FOLDER_ID")

    sched = BlockingScheduler(timezone=TZ)

    for h in (9, 13, 18, 21):
        sched.add_job(
            run_job,
            trigger=CronTrigger(hour=h, minute=30, timezone=TZ),
            coalesce=True,
            max_instances=1,
            misfire_grace_time=900,
        )

    log.info("Scheduler started TZ=%s times=09:30,13:30,18:30,21:30", TZ)

    if RUN_NOW:
        log.info("RUN_NOW=1 -> running one job immediately")
        run_job()

    sched.start()


if __name__ == "__main__":
    main()
