import copy
import random
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

summarizer = pipeline("summarization")
qa_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl", tokenizer="t5-base", use_fast=False)

def expand_data(dataset, config):
    """
    Expands the given dataset based on the specified config.
    Config includes:
    - A boolean adjacency matrix indicating allowed type transformations.
    - Specific methods for each transformation case.
    """
    sz = dataset.__len__()
    for id in range(sz):
        qa_pair = dataset.get(id)
        current_type = qa_pair.classify_id()

        for target_type in range(1, len(config["adjacency_matrix"][current_type])):
            allowed = config["adjacency_matrix"][current_type][target_type]
            if allowed:
                # If the target type is a strict subset of the current type, just remove fields
                if is_subset_type(current_type, target_type):
                    dataset.add(reduce_to_type(qa_pair, target_type))
                    continue
                
                # Otherwise, apply specific expansion methods for the transformation
                expanded_entry = expand_case_by_case(qa_pair, current_type, target_type, config)
                for entry in expanded_entry:
                    entry.links.append(qa_pair.id)  # Add a link to the original entry
                    dataset.add(entry)

    
    return dataset  # Append new entries to the dataset

def is_subset_type(current_type, target_type):
    """Checks if the target type is a strict subset of the current type."""
    return (current_type & target_type) == target_type

def reduce_to_type(qa_pair, target_type):
    """Removes extra fields to match the target type."""
    new_entry = copy.deepcopy(qa_pair)
    new_entry.links.append(qa_pair.id)  # Add a link to the original entry
    if not (target_type & 1):  # If C is not included in target
        new_entry.context = None
    if not (target_type & 2):  # If Q is not included in target
        new_entry.question = None
    if not (target_type & 4):  # If A is not included in target
        new_entry.answer = None
    return new_entry

def expand_case_by_case(qa_pair, current_type, target_type, config):
    """Applies specific expansion methods based on the transformation type."""
    # Placeholder for actual implementations based on research methods
    if current_type == 1 and target_type == 7:  # C -> C+Q+A
        return generator_QA_pair_from_context(qa_pair, config)
    if current_type == 1 and target_type == 3:  # C -> C+Q
        return generator_Q_pair_from_context(qa_pair, config)
    return []  # Skip if not implemented yet

# 生成完整 QA 配對
def generator_QA_pair_from_context(qa_pair, config):
    method = {
        "generate_fill_in_blank": generate_fill_in_blank
    }
    new_qa_pairs = []
    for key, value in method.items():
        if config.get(key):
            new_qa_pairs += value(copy.deepcopy(qa_pair))
    return new_qa_pairs

# 生成只包含問題的 QA 配對
def generator_Q_pair_from_context(qa_pair, config):
    method = {
        "extract_key_sentences": extract_key_sentences,
        "generate_summary_qa": generate_summary_qa
    }
    new_qa_pairs = []
    for key, value in method.items():
        if config.get(key):
            new_qa_pairs += value(copy.deepcopy(qa_pair))
    return new_qa_pairs

def extract_key_sentences(qa_pair):
    '''
    生成問答對，問題是關鍵句子的意思
    '''
    sentences =  qa_pair.context_keywords
    if not sentences:
        return []
    key_sentence = random.choice(sentences)
    question = f"根據內容，{key_sentence} 是什麼意思？"
    qa_pair.set_question(question)
    return [qa_pair]

# 方法 1: 生成摘要型問答
def generate_summary_qa(qa_pair):
    summarized_text = summarizer(qa_pair.context, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
    question = f"根據內容，總結的關鍵資訊是什麼？"
    qa_pair.set_question(question)
    qa_pair.set_answer(summarized_text)
    return [qa_pair]

# 方法 2: 生成類比問答
def generate_analogy_qa(qa_pair):
    keywords = qa_pair.context_keywords
    if len(keywords) < 2:
        return []
    analogy = f"{keywords[0]} 和 {keywords[1]} 有什麼相似之處？"
    qa_pair.set_question(analogy)
    qa_pair.set_answer("兩者都涉及...")
    return [qa_pair]


# 方法 5: 生成填空題
def generate_fill_in_blank(qa_pair):
    context = qa_pair.context
    words = word_tokenize(context)
    words = [w for w in words if w.isalnum() and w.lower() not in stopwords.words("english")]
    if not words:
        return []
    blank_word = random.choice(words)
    question = context.replace(blank_word, "____")
    qa_pair.set_question(question)
    qa_pair.set_answer(blank_word)
    return [qa_pair]
