from typing import List
import re
from QA_GEN.base import QAPair
from QA_GEN.llms.ollama_chat import OllamaChat
from QA_GEN.method_register import Method

@Method(
    name="paq_answer_extraction", 
    description="Extract k candidate answer spans from context using PAQ-style answer extraction. Creates multiple QAPairs with the same context but different extracted answers, leaving questions empty for later generation.",
    applicable_stages=["data_expansion"],
    use_LLM=True,
    complexity="O(n) where n is the number of context pairs; LLM calls may impact performance"
)
def paq_answer_extraction(qa_pairs: List[QAPair], config, k: int = 8, llm = None) -> List[QAPair]:
    """
    Extract k candidate answer spans from context using LLM-based extraction.
    
    Args:
        qa_pairs: List of QAPair objects (should contain context-only pairs)
        config: Configuration object
        k: Number of candidate answers to extract per context (default: 3)
    
    Returns:
        List of new QAPair objects with extracted answers
    """
    extracted_pairs = []
    
    for context_pair in qa_pairs:
        if context_pair.classify_id() == 1:  # Context-only type (Document)
            if llm is None:
                llm = OllamaChat()
            
            # Prompt for extracting SHORT candidate answer spans
            prompt = f"""
You are an AI specialized in extracting SHORT answer spans from given contexts.
Your task is to identify {k} different SHORT text spans that could serve as concise answers to potential questions.

CRITICAL REQUIREMENTS:
1. Extract exactly {k} different SHORT spans from the context
2. Focus on: proper nouns, dates, numbers, key terms, specific facts
3. Avoid complete sentences or long phrases
4. Choose the most informative and precise spans
5. Examples of good spans: "2023", "John Smith", "machine learning", "25%", "New York"

Format your response as numbered list:
1. <span 1>
2. <span 2>
3. <span 3>
...

Context: {context_pair.context[:2000]}
"""
            
            try:
                response_text, response_info = llm(prompt=prompt)
                
                # Parse the numbered list response
                lines = response_text.strip().split('\n')
                extracted_answers = []
                
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                        # Remove the number prefix (e.g., "1. ", "2. ", etc.)
                        answer = re.sub(r'^\d+\.\s*', '', line).strip()
                        
                        # Filter for short spans (enforce length constraint)
                        if answer and len(answer.split()) <= 8:  # Max 8 words
                            # Remove quotes if present
                            answer = answer.strip('"\'')
                            if answer:
                                extracted_answers.append(answer)
                
                # Create new QAPairs for each extracted answer
                for i, answer in enumerate(extracted_answers[:k]):  # Limit to k answers
                    new_pair = QAPair(
                        context=context_pair.context,
                        question=None,  # Question remains empty
                        answer=answer
                    )
                    
                    extracted_pairs.append(new_pair)
                
            except Exception as e:
                continue

    return extracted_pairs