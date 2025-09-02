from typing import List
import re
from QA_GEN.base import QAPair
from QA_GEN.llms.ollama_chat import OllamaChat
from QA_GEN.method_register import Method

@Method(
    name="cross_domain_fusion",
    description="Fuse two moderately related QA pairs to create a new hybrid QA pair that bridges both domains",
    applicable_stages=["data_fusion"],
    use_LLM=True
)
def cross_domain_fusion(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Create a new QA pair by fusing knowledge from two moderately related QA pairs.
    This method works best when the input pairs have some conceptual overlap but
    are from different domains or contexts.
    
    Args:
        qa_pairs (List[QAPair]): List of QA pairs to fuse (ideally 2 pairs)
        
    Returns:
        List[QAPair]: Newly generated hybrid QA pair
    """
    # Need exactly 2 QA pairs for this fusion method
    if len(qa_pairs) != 2:
        print("Cross-domain fusion requires exactly 2 QA pairs")
        return []
    
    pair1, pair2 = qa_pairs
    
    # Create prompt for LLM
    llm = OllamaChat()
    prompt = f"""
You are an expert at creating content that bridges multiple domains or topics.
I'll show you two different QA pairs that have some moderate relationship, and I want you to create a new, 
original QA pair that creatively combines elements from both sources.

First QA Pair:
Context: {pair1.context}
Question: {pair1.question}
Answer: {pair1.answer}

Second QA Pair:
Context: {pair2.context}
Question: {pair2.question}
Answer: {pair2.answer}

Create a new QA pair that bridges these two topics or domains. The new QA pair should:
1. Identify a conceptual link or relationship between these two areas
2. Create something novel that couldn't be derived from either source alone
3. Be factually accurate

Please provide your response in this format:
Context: (a bridge context that connects elements from both sources)
Question: (your original question that spans both domains)
Answer: (your original answer that incorporates knowledge from both sources)
"""

    # Generate new QA pair
    response_text, response_info = llm(prompt=prompt)
    
    # Parse the response
    context_match = re.search(r"Context:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    question_match = re.search(r"Question:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    answer_match = re.search(r"Answer:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    
    if context_match and question_match and answer_match:
        new_context = context_match.group(1).strip()
        new_question = question_match.group(1).strip()
        new_answer = answer_match.group(1).strip()
        
        # Create a new QA pair
        new_pair = QAPair(
            question=new_question,
            answer=new_answer,
            context=new_context
        )
        
        return [new_pair]
    else:
        print("Failed to parse the LLM response for cross-domain fusion")
        return []