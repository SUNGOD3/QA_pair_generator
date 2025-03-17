import copy
import random
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

def generator_QA_pair_from_context(qa_pair, config):
    new_qa_pairs = []
    if config["extract_key_sentences"]:
        new_qa_pairs += extract_key_sentences(copy.deepcopy(qa_pair))
    #if config["generate_summary_question"]:
    #    new_qa_pairs += generate_summary_question(qa_pair)
    #if config["generate_entity_based_QA"]:
    #    new_qa_pairs += generate_entity_based_QA(qa_pair)
    #if config["generate_cause_effect_QA"]:
    #    new_qa_pairs += generate_cause_effect_QA(qa_pair)
    return new_qa_pairs

def generator_Q_pair_from_context(qa_pair, config):
    new_qa_pairs = []
    if config["generate_fill_in_blank"]:
        new_qa_pairs += generate_fill_in_blank(copy.deepcopy(qa_pair))
    return new_qa_pairs

def extract_key_sentences(qa_pair):
    sentences =  qa_pair.context_keywords
    if not sentences:
        return []
    key_sentence = random.choice(sentences)
    question = f"根據內容，{key_sentence} 是什麼意思？"
    qa_pair.set_question(question)
    answer = key_sentence # TODO: Implement actual keyword extraction
    qa_pair.set_answer(answer)
    return [qa_pair]

def generate_fill_in_blank(qa_pair):
    context = qa_pair.context
    words = word_tokenize(context)
    words = [w for w in words if w.isalnum() and w.lower() not in stopwords.words("english")]
    if not words:
        return []
    blank_word = random.choice(words)
    question = context.replace(blank_word, "____")
    qa_pair.set_question(question)
    return [qa_pair]

#def generate_summary_question(context, sentences):
#    if len(sentences) < 2:
#        return []
#    summary = sentences[0]  # 取第一句作為摘要
#    question = f"這段話的主要內容是什麼？"


#def generate_entity_based_QA(context, sentences):
#    entities = re.findall(r"\b[A-Z][a-z]+\b", context)
#    if not entities:
#        return []
#    entity = random.choice(entities)
#    question = f"{entity} 是什麼？"
#    return [QAPair(question, entity, context)]

# 方法5: 因果關係問答
#def generate_cause_effect_QA(context, sentences):
#    cause_effect_patterns = [
#        r"(.*)因為(.*)",
#        r"(.*)導致(.*)",
#        r"(.*)所以(.*)",
#        r"(.*)結果是(.*)"
#    ]
#    for sentence in sentences:
##        for pattern in cause_effect_patterns:
 #           match = re.search(pattern, sentence)
 #           if match:
 #               cause, effect = match.groups()
 ##               question = f"為什麼 {effect.strip()} ？"
  #  return []