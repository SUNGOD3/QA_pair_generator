from typing import List
from QA_GEN.base import QAPair
from QA_GEN.method_register import Method

@Method(
    name="t5_question_augmentation",
    description="Augment QA pairs by generating paraphrased questions using T5 model",
    applicable_stages=["data_augmentation"],
    use_LLM=False,
    use_docker=False,
)
def t5_question_augmentation(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Augment QA pairs by generating paraphrased versions of questions using T5.
    
    Args:
        qa_pairs (List[QAPair]): List of QA pairs to augment
        config: Configuration object with k parameter for number of paraphrases
    
    Returns:
        List[QAPair]: List of original and augmented QA pairs
    """
    k = getattr(config, 'k', 2)
    
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
        model = T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_paraphraser")
    except:
        return []
    
    augmented_pairs = []
    
    for qa_pair in qa_pairs:
        if not qa_pair.question or not qa_pair.question.strip():
            continue
            
        try:
            input_text = f"paraphrase: {qa_pair.question}"
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=k + 1,
                num_return_sequences=min(k, 3),
                temperature=0.8,
                do_sample=True,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id
            )
            
            for output in outputs:
                paraphrase = tokenizer.decode(output, skip_special_tokens=True)
                norm_q = ''.join(filter(str.isalnum, qa_pair.question.lower()))
                norm_p = ''.join(filter(str.isalnum, paraphrase.lower()))
                if norm_q != norm_p and paraphrase.strip():
                    augmented_pair = QAPair(
                        question=paraphrase,
                        answer=qa_pair.answer,
                        context=qa_pair.context
                    )
                    augmented_pairs.append(augmented_pair)
                    if len([p for p in augmented_pairs if p.question == paraphrase]) >= k:
                        break
        except:
            continue
    
    return augmented_pairs