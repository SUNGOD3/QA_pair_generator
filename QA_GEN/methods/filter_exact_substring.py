from typing import List
from QA_GEN.base import QAPair
from QA_GEN.method_register import Method

@Method(
    name="filter_exact_substring", 
    description="Filter out QA pairs where one contains a substantial exact substring of another",
    applicable_stages=["data_filter"],
    use_LLM=False
)
def filter_exact_substring(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Filters out QA pairs where one contains a substantial exact substring of another.
    For each match, the longer QA pair is removed.
    
    Args:
        qa_pairs: A list of QAPair objects to filter
        config: Configuration dictionary containing parameter 'k' for minimum substring length
    
    Returns:
        List[QAPair]: Filtered list of QAPair objects with redundant pairs removed
    """
    if len(qa_pairs) <= 1:
        return qa_pairs
    
    # Get minimum substring length from config, default to 50 characters
    min_substring_length = config.get('k', 50)
    
    # Create a copy to avoid modifying the original list during iteration
    pairs_to_remove = set()
    
    # Helper function to check for substring match
    def check_substring_match(text1: str, text2: str) -> bool:
        """Check if one text contains a substantial substring of the other."""
        if not text1 or not text2:
            return False
            
        # Check if text1 contains text2 as a substantial substring
        if text1 != text2 and len(text2) >= min_substring_length and text2 in text1:
            return True
            
        # Check if text2 contains text1 as a substantial substring
        if text1 != text2 and len(text1) >= min_substring_length and text1 in text2:
            return True
            
        return False
    
    # Compare all pairs
    for i in range(len(qa_pairs)):
        if i in pairs_to_remove:
            continue
            
        for j in range(i+1, len(qa_pairs)):
            if j in pairs_to_remove:
                continue
                
            pair1, pair2 = qa_pairs[i], qa_pairs[j]
            
            # Extract all text fields from each pair
            texts1 = [pair1.context or "", pair1.question or "", pair1.answer or ""]
            texts2 = [pair2.context or "", pair2.question or "", pair2.answer or ""]
            
            # Check each combination of text fields for substring matches
            found_match = False
            for t1 in texts1:
                for t2 in texts2:
                    if check_substring_match(t1, t2):
                        found_match = True
                        break
                if found_match:
                    break
            
            # If a match is found, remove the longer pair
            if found_match:
                # Calculate total length of each pair
                len1 = sum(len(t) for t in texts1)
                len2 = sum(len(t) for t in texts2)
                
                # Remove the longer pair (prioritize keeping shorter content)
                if len1 > len2:
                    pairs_to_remove.add(i)
                else:
                    pairs_to_remove.add(j)
    
    # Create final filtered list
    filtered_pairs = [pair for idx, pair in enumerate(qa_pairs) if idx not in pairs_to_remove]
    
    return filtered_pairs