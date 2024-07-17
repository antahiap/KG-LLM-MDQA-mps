from pathlib import Path
from itertools import groupby
import spacy
from typing import List, Dict, Any, Tuple, Optional, Union
nlp = spacy.load('en_core_web_sm')


def get_element_group_closure(part: int) -> str:
    def get_element_group(element: dict):
        parts = Path(element['Path']).parts
        if part >= len(parts):
            return ''
        return parts[part]

    return get_element_group


def gather_text(elements: List[dict], sep: str = ' ') -> str:
    return sep.join([e.get('Text', '') for e in elements])


def table_to_markdown(table: List[dict], level: int) -> str:
    tr_getter = get_element_group_closure(level + 1)
    td_getter = get_element_group_closure(level + 2)

    sentences = ''
    for part, cells in groupby(table, tr_getter):
        row = []
        row_text = ''
        if part.startswith('TR'):
            for _, cell in groupby(cells, td_getter):
                row += [gather_text(cell, ' ')]

            row_text = '|'.join(row)
            row_text = '|' + row_text + '|'

        sentences += row_text + '\n'

    return sentences


def sentence_tokenize_block(
    block_type: str,
    block_elements: List[dict],
    level: int
) -> List[str]:
    if block_type.startswith('Sect'):
        # recursively descend the `Sect` tree
        sentences = []
        level_getter = get_element_group_closure(level + 1)
        for group, elements in groupby(block_elements, level_getter):
            sentences += sentence_tokenize_block(group, elements, level=level + 1)
        return sentences

    if block_type.startswith('Title'):
        return [gather_text(block_elements)]

    if block_type.startswith('P') or block_type.startswith('Aside') or block_type.startswith('Footnote'):
        text = gather_text(block_elements)
        return [sent.text.strip() for sent in nlp(text).sents]

    if block_type.startswith('H'):
        return [gather_text(block_elements).strip()]

    if block_type.startswith('Figure') or block_type.startswith('Watermark'):
        return []

    if block_type.startswith('L'):
        li_getter = get_element_group_closure(level + 1)
        sentences = []
        for _, li in groupby(block_elements, li_getter):
            current_label = ''
            for li_element in li:
                element_type = Path(li_element.get('Path', '')).stem

                # if element_type == 'Lbl':
                #    current_label = li_element.get('Text', '').strip()
                if element_type == 'LBody':
                    text = li_element.get('Text', '')
                    snts = [sent.text.strip() for sent in nlp(text).sents]
                    snts = [current_label + ' ' + snts[0]] + snts[1:]
                    sentences.extend(snts)
                    current_label = ''
        return sentences

    if block_type.startswith('TOC'):
        toci_getter = get_element_group_closure(level + 1)
        sentences = []
        for _, toci in groupby(block_elements, toci_getter):
            sentences.append(gather_text(toci))
        return sentences

    if block_type.startswith('Table'):
        return [table_to_markdown(block_elements, level)]

    return []


def sentence_tokenize_extract(extract: dict) -> List[str]:
    sentences = []

    l2_getter = get_element_group_closure(2)
    for group, elements in groupby(extract['elements'], l2_getter):
        sentences += sentence_tokenize_block(group, elements, level=2)

    return sentences