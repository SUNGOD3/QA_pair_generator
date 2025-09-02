from QA_GEN.llms.oai_chat import OpenAIChat
from QA_GEN.method_register import Method

@Method(
    name="generate_summary_qa",
    description="Generate summary-type QA pairs, where the question is about the summary of the content",
    applicable_stages=["data_expansion"],
    use_LLM=True
)
def generate_summary_qa(qa_pairs, config):
    rt_pairs = []
    llm = OpenAIChat()
    question = f"Based on the content, what is the key information summarized?"
    for qa_pair in qa_pairs:
        qa_pair.set_question(question)
        response_text, response_info = llm(prompt=qa_pair.context + '\n' + question)
        summarized_text = response_text.strip()
        qa_pair.set_answer(summarized_text)
        rt_pairs.append(qa_pair)
    return rt_pairs