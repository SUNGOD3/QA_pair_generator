import random
from QA_GEN.method_register import Method

@Method(
    name="key_sentences_question",
    description="Generate only questions based on key sentences",
    applicable_stages=["data_expansion"],
    use_LLM=False
)
def key_sentences_question(qa_pairs, config):
    rt_pairs = []
    for qa_pair in qa_pairs:
        sentences =  qa_pair.context_keywords
        if not sentences:
            return []
        key_sentence = random.choice(sentences)
        question = f"Based on the content, what does '{key_sentence}' mean?"
        qa_pair.set_question(question)
        rt_pairs.append(qa_pair)
    return rt_pairs