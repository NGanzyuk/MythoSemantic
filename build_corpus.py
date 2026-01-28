#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import requests
import unicodedata
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOWNLOAD_LIST_FILE = "download_list.json"
CORPUS_DIR = Path("corpus")
METADATA_FILE = CORPUS_DIR / "corpus_metadata.json"
CATALOG_FILE = CORPUS_DIR / "corpus_catalog.csv"


def load_download_list():
    with open(DOWNLOAD_LIST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def download_file(url: str) -> bytes:
    logger.info(f"Загрузка: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def html_to_text(html_content: bytes) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def build_corpus():
    ensure_dir(CORPUS_DIR)
    metadata = []
    catalog_rows = []

    download_list = load_download_list()

    for item in download_list:
        tid = item["id"]
        tradition = item["tradition"]
        lang = item["language"]
        ftype = item["type"]
        url = item["url"]

        tradition_path = tradition.replace('/', '_').replace(' ', '_')
        folder = CORPUS_DIR / tradition_path / tid
        ensure_dir(folder)
        filename = folder / f"{tid}.txt"

        try:
            data = download_file(url)

            if b"<html" in data[:150].lower():
                logger.debug(f"{tid}: Обнаружен HTML, преобразуем в текст")
                text = html_to_text(data)
                text = normalize_text(text)
                data = text.encode("utf-8")

            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.write_bytes(data)

            h = md5(data)

            meta = {
                "id": tid,
                "tradition": tradition,
                "language": lang,
                "type": ftype,
                "url": url,
                "date_downloaded": datetime.utcnow().isoformat(),
                "md5": h,
                "path": str(filename.resolve())
            }
            metadata.append(meta)
            catalog_rows.append([tid, tradition, lang, ftype, str(filename.resolve()), url, h])

            logger.info(f"Успешно сохранено: {tid}")

        except Exception as e:
            logger.error(f"Не удалось обработать {tid}: {e}")
            continue

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(CATALOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "tradition", "language", "type", "path", "url", "md5"])
        writer.writerows(catalog_rows)

    logger.info("Сборка корпуса завершена.")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
