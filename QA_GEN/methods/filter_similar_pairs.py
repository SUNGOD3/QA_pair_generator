from typing import List
from QA_GEN.base import QAPair
from QA_GEN.llms.ollama_chat import OllamaChat
from QA_GEN.method_register import Method
import random

@Method(
    name="filter_similar_pairs", 
    description="Filter out similar QA pairs by randomly sampling and comparing pairs using LLM",
    applicable_stages=["data_filter"],
    use_LLM=True,
)
def filter_similar_pairs(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Filters out similar QA pairs by randomly sampling K pairs and comparing them using LLM.
    For each comparison, if pairs are found to be similar, the shorter one is removed.
    
    Args:
        qa_pairs: A list of QAPair objects to filter
        config: Configuration dictionary containing parameters like 'k' for sampling times
    
    Returns:
        List[QAPair]: Filtered list of QAPair objects with similar pairs removed
    """
    if len(qa_pairs) <= 1:
        return qa_pairs
    
    # Get sampling parameter K from config
    k = config.get('k', min(10, len(qa_pairs) * (len(qa_pairs) - 1) // 2))
    
    # Create a copy to avoid modifying the original list during iteration
    remaining_pairs = qa_pairs.copy()
    pairs_to_remove = set()
    
    # Keep track of already compared pairs to avoid duplicates
    compared_pairs = set()
    
    # Initialize LLM
    llm = OllamaChat()
    
    # Sample K times
    sample_count = 0
    max_attempts = k * 2  # To avoid infinite loops if we run out of unique pairs
    attempt_count = 0
    
    while sample_count < k and attempt_count < max_attempts and len(remaining_pairs) > 1:
        attempt_count += 1
        
        # Get indices of two random pairs that haven't been removed yet
        valid_indices = [i for i, pair in enumerate(remaining_pairs) if i not in pairs_to_remove]
        if len(valid_indices) < 2:
            break
        
        idx1, idx2 = random.sample(valid_indices, 2)
        pair1, pair2 = remaining_pairs[idx1], remaining_pairs[idx2]
        
        # Create a unique identifier for this comparison
        pair_id = tuple(sorted([pair1.id, pair2.id]))
        
        # Skip if we've already compared these pairs
        if pair_id in compared_pairs:
            continue
        
        # Mark this pair as compared
        compared_pairs.add(pair_id)
        sample_count += 1
        
        # Prepare prompt to check similarity
        prompt = f"""
Compare these two QA pairs and determine if they are semantically similar or duplicates.
Answer with only 'YES' if they are similar enough that one should be removed, or 'NO' if they are sufficiently different.

QA Pair 1:
Context: {pair1.context or "N/A"}
Question: {pair1.question or "N/A"}
Answer: {pair1.answer or "N/A"}

QA Pair 2:
Context: {pair2.context or "N/A"}
Question: {pair2.question or "N/A"}
Answer: {pair2.answer or "N/A"}

Are these QA pairs similar or duplicates? (YES/NO)
"""
        
        # Ask LLM if pairs are similar
        response, _ = llm(prompt=prompt)
        
        # Check if response indicates similarity
        if "YES" in response.upper():
            # Calculate length of each pair (total characters in context+question+answer)
            len1 = len((pair1.context or "") + (pair1.question or "") + (pair1.answer or ""))
            len2 = len((pair2.context or "") + (pair2.question or "") + (pair2.answer or ""))
            
            # Remove the shorter pair
            if len1 <= len2:
                pairs_to_remove.add(idx1)
            else:
                pairs_to_remove.add(idx2)
    
    # Create final filtered list
    filtered_pairs = [pair for i, pair in enumerate(remaining_pairs) if i not in pairs_to_remove]
    
    return filtered_pairs