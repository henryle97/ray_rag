from pathlib import Path
from typing import List

from bs4 import BeautifulSoup, NavigableString

import rag.config as config
from rag.schemas import Record, Section


def extract_sections(record: dict | Record) -> List[dict]:
    if isinstance(record, dict):
        record = Record(**record)
    with open(record.path, "r", encoding="utf8") as html_file:
        soup = BeautifulSoup(html_file, "html.parser")
    sections = soup.find_all("section")
    section_list = []
    for section in sections:
        section_id = section.get("id")
        section_text = extract_text_from_section(section)
        if section_id:
            url = path_to_uri(path=record.path)
            section_obj = Section(source=url, text=section_text)
            section_list.append(section_obj.to_dict())
    return section_list


def extract_text_from_section(section: BeautifulSoup):
    texts = []
    for element in section.children:
        if isinstance(element, NavigableString):
            if element.strip():
                texts.append(element)
        elif element.name == "section":
            continue
        else:
            texts.append(element.get_text().strip())
    return "\n".join(texts)


def path_to_uri(path: str, scheme="https://", domain: str = "docs.ray.io"):
    return scheme + domain + str(path).split(domain)[-1]

def fetch_text(uri: str) -> str:
    """
    uri = "https://docs.ray.io/en/master/data/transforming-data.html#configuring-batch-format"
    uri = "https://docs.ray.io/en/master/data/transforming-data.html
    """
    url, anchor = uri.split("#") if "#" in uri else (uri, None)
    file_path = Path(config.EFS_DIR, url.split("https://")[-1])
    
    if not file_path.exists():
        print(f"File not found at {file_path}")
        return "" 
    with open(file_path, "r", encoding="utf8") as html_file:
        html_content = html_file.read()
    soup = BeautifulSoup(markup=html_content, features="html.parser")
    if anchor:
        target_element = soup.find(id=anchor)
        if target_element:
            text = target_element.get_text()
        else:
            return fetch_text(uri=url)
    else:
        text = soup.get_text()
    
    return text