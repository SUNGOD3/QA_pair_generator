from QA_GEN.method_register import Method
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random

@Method(
    name="generate_fill_in_blank",
    description="Generate fill-in-the-blank type QA pairs",
    applicable_stages=["data_expansion"],
    use_LLM=False
)
def generate_fill_in_blank(qa_pairs, config):
    rt_pairs = []
    for qa_pair in qa_pairs:
        context = qa_pair.context
        words = word_tokenize(context)
        words = [w for w in words if w.isalnum() and w.lower() not in stopwords.words("english")]
        if not words:
            return []
        blank_word = random.choice(words)
        question = context.replace(blank_word, "____")
        qa_pair.set_question(question)
        qa_pair.set_answer(blank_word)
        rt_pairs.append(qa_pair)
    return rt_pairs

print("Methods registered:")
for name, method in Method.get_methods().items():
    print(f" - {name}: {method['description']}")