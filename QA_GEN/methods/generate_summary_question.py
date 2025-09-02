from QA_GEN.method_register import Method

@Method(
    name="generate_summary_question",
    description="Generate summary questions, where the question is about the summary of the content",
    applicable_stages=["data_expansion"],
    use_LLM=False
)
def generate_summary_question(qa_pairs, config):
    rt_pairs = []
    for qa_pair in qa_pairs:
        question = f"Based on the content, what is the key information summarized?"
        qa_pair.set_question(question)
        rt_pairs.append(qa_pair)
    return rt_pairs
