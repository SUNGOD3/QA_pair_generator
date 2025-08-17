import json
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from QA_GEN.llms.ollama_chat import OllamaChat
import random

@dataclass
class ScoredItem:
    context: str
    question: str
    answer: str
    score: float
    raw_response: str

class DatasetScorer:
    def __init__(self, model: str):
        self.llm_model = model
        self.llm = OllamaChat(model_name=model)
        
    def load_jsonl(self, filepath: str) -> List[Dict]:
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def create_scoring_prompt(self, context: str, answer: str, question: str) -> str:
        prompt = f"""You are an expert question quality evaluator. Please evaluate the quality of the given question based on the provided context and correct answer.

**Evaluation Criteria (1-10 scale):**
- **Precision (3 points)**: Does the question precisely guide to the unique correct answer? Does it avoid ambiguity?
- **Clarity (2 points)**: Is the question clear, well-structured, and easy to understand?
- **Appropriateness (2 points)**: Is the question appropriate for the content depth and scope of the given context?
- **Answerability (2 points)**: Can the question be answered directly from the context without external knowledge?
- **Specificity (1 point)**: Does the question target the specific information rather than being too general?

**Scoring Guide:**
- 9-10: Excellent - Perfect question with no issues
- 7-8: Good - Minor issues that don't significantly impact quality
- 5-6: Average - Some issues but still functional
- 3-4: Poor - Significant issues affecting quality
- 1-2: Very Poor - Major problems, barely functional

**Context:**
{context}

**Correct Answer:** {answer}

**Question to Evaluate:** {question}

**Instructions:**
1. Evaluate the question based on the criteria above
2. Consider how well this question would lead someone to the correct answer
3. Provide a score from 1 to 10 (can use decimals like 7.5)

**Response Format:**
Score: [Your numerical score]
Brief Reasoning: [1-2 sentences explaining your score]

**Your Evaluation:**"""
        return prompt
    
    def parse_score_response(self, response: str) -> tuple[float, str]:
        try:
            lines = response.strip().split('\n')
            score = None
            reasoning = ""
            
            for line in lines:
                line = line.strip()
                if line.lower().startswith('score:'):
                    score_text = line.split(':', 1)[1].strip()
                    import re
                    score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                    if score_match:
                        score = float(score_match.group(1))
                        score = max(1.0, min(10.0, score))
                elif line.lower().startswith('brief reasoning:') or line.lower().startswith('reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            # 如果沒有找到分數，嘗試從整個回應中提取
            if score is None:
                import re
                score_matches = re.findall(r'(\d+(?:\.\d+)?)', response)
                if score_matches:
                    potential_score = float(score_matches[0])
                    if 1.0 <= potential_score <= 10.0:
                        score = potential_score
            
            if score is None:
                score = -1.0
                reasoning = "Failed to parse score, assigned neutral score"
            
            return score, reasoning
            
        except Exception as e:
            return -1.0, f"Parsing failed: {str(e)}"
    
    def score_item(self, context: str, answer: str, question: str) -> tuple[float, str]:
        prompt = self.create_scoring_prompt(context, answer, question)
        
        try:
            response_text, response_info = self.llm(prompt=prompt)
            score, reasoning = self.parse_score_response(response_text)
            return score, response_text
            
        except Exception as e:
            return -1.0, f"LLM call failed: {str(e)}"
    
    def score_dataset(self, dataset_path: str, output_path: Optional[str] = None) -> Dict:
        dataset = self.load_jsonl(dataset_path)
        dataset_name = ["PAQ", "llama3.2", "qwen3"]
        mean_values = {} # name: [scores, failures]

        for name in dataset_name:
            scored_items = []
            scores = []
            fail = 0
            for i, item in enumerate(dataset, 1):
                
                score, raw_response = self.score_item(
                    context=item['context'],
                    answer=item['answer'],
                    question=item[name]
                )
                
                scored_item = ScoredItem(
                    context=item['context'],
                    question=item[name],
                    answer=item['answer'],
                    score=score,
                    raw_response=raw_response
                )
                scored_items.append(scored_item)
                if score < 0:
                    fail += 1
                    continue
                scores.append(score)
            
            results = {
                'dataset_path': dataset_path,
                'total_items': len(scored_items),
                'average_score': statistics.mean(scores),
                'median_score': statistics.median(scores),
                'std_deviation': statistics.stdev(scores) if len(scores) > 1 else 0,
                'min_score': min(scores),
                'max_score': max(scores),
                'score_distribution': self._get_score_distribution(scores),
                'scored_items': []
            }
            
            count = 0
            for item in scored_items:
                results['scored_items'].append({
                    'id': count,
                    'dataset': name,
                    'score': item.score,
                    'raw_response': item.raw_response
                })
                count += 1
            if mean_values.get(name) is None:
                mean_values[name] = [[results['average_score'], fail]]
            else:
                mean_values[name].append([results['average_score'], fail])
            
        print("\n=== Evaluation Results ===")
        print(mean_values)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def _get_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        distribution = {
            "9.0-10.0 (Excellent)": 0,
            "7.0-8.9 (Good)": 0,
            "5.0-6.9 (Average)": 0,
            "3.0-4.9 (Poor)": 0,
            "1.0-2.9 (Very Poor)": 0,
            "-1.0 (Invalid)": 0
        }
        
        for score in scores:
            if 9.0 <= score <= 10.0:
                distribution["9.0-10.0 (Excellent)"] += 1
            elif 7.0 <= score < 9.0:
                distribution["7.0-8.9 (Good)"] += 1
            elif 5.0 <= score < 7.0:
                distribution["5.0-6.9 (Average)"] += 1
            elif 3.0 <= score < 5.0:
                distribution["3.0-4.9 (Poor)"] += 1
            elif 1.0 <= score < 3.0:
                distribution["1.0-2.9 (Very Poor)"] += 1
            else:
                distribution["-1.0 (Invalid)"] += 1
        return distribution



@dataclass
class DataItem:
    context: str
    question: str
    answer: str
    source: str  # 'A' or 'B'

class QuestionEvaluator:
    def __init__(self, model: str):
        self.llm = OllamaChat(model_name=model)

    def load_jsonl(self, filepath: str) -> List[Dict]:
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def find_intersection(self, dataset: List[Dict], dataset_name_a: str, dataset_name_b: str) -> List[Tuple[DataItem, DataItem]]:
        pairs = []
        for item in dataset:
            item_a = DataItem(
                context=item['context'],
                question=item[dataset_name_a],
                answer=item['answer'],
                source='A'
            )
            item_b = DataItem(
                context=item['context'],
                question=item[dataset_name_b],
                answer=item['answer'],
                source='B'
            )
            pairs.append((item_a, item_b))
        
        return pairs
    
    def create_evaluation_prompt(self, context: str, answer: str, question1: str, question2: str) -> str:
        prompt = f"""You are a question quality evaluation expert. Please evaluate the quality of the following two questions based on detailed criteria.

**Evaluation Criteria (aligned with 1-10 scale standards):**
- **Precision (3 points)**: Does the question precisely guide to the unique correct answer? Does it avoid ambiguity?
- **Clarity (2 points)**: Is the question clear, well-structured, and easy to understand?
- **Appropriateness (2 points)**: Is the question appropriate for the content depth and scope of the given context?
- **Answerability (2 points)**: Can the question be answered directly from the context without external knowledge?
- **Specificity (1 point)**: Does the question target the specific information rather than being too general?

**Quality Standards:**
- Excellent (9-10): Perfect question with no issues
- Good (7-8): Minor issues that don't significantly impact quality  
- Average (5-6): Some issues but still functional
- Poor (3-4): Significant issues affecting quality
- Very Poor (1-2): Major problems, barely functional

**Context:** {context}

**Correct Answer:** {answer}

**Question 1:** {question1}

**Question 2:** {question2}

**Instructions:**
1. Evaluate both questions based on the criteria above
2. Consider how well each question would lead someone to the correct answer
3. Compare their overall quality based on the 1-10 scale standards
4. Choose the question that better meets the evaluation criteria

Please choose the better quality question. If both questions are of very similar quality (within 0.5 points), choose "Tie".

**Response Format:**
Only respond with one of the following:
- Question 1
- Question 2  
- Tie

**Your choice:**"""
        return prompt
    
    def parse_llm_response(self, response: str) -> str:
        response = response.strip().lower()
        
        if "question 1" in response or "question1" in response or response.startswith("1"):
            return "question1"
        elif "question 2" in response or "question2" in response or response.startswith("2"):
            return "question2"
        elif "tie" in response:
            return "tie"
        else:
            return "fail"

    def evaluate_pair(self, item_a: DataItem, item_b: DataItem) -> str:
        if random.choice([True, False]):
            question1, question2 = item_a.question, item_b.question
            first_source = 'A'
        else:
            question1, question2 = item_b.question, item_a.question
            first_source = 'B'
        
        prompt = self.create_evaluation_prompt(
            context=item_a.context,
            answer=item_a.answer, 
            question1=question1,
            question2=question2
        )
        
        try:
            response_text, response_info = self.llm(prompt=prompt)
            result = self.parse_llm_response(response_text)
            
            if result == "question1":
                return 'A' if first_source == 'A' else 'B'
            elif result == "question2":
                return 'B' if first_source == 'A' else 'A'
            elif result == "tie":
                return 'tie'
            else:
                return 'fail'
                
        except Exception as e:
            print(e)
            return 'tie'
    
    def run_evaluation(self, dataset_path: str, dataset_name_a: str, dataset_name_b: str, output_path: Optional[str] = None) -> Dict:
        dataset = self.load_jsonl(dataset_path)
        pairs = self.find_intersection(dataset, dataset_name_a, dataset_name_b)
        
        results = {
            'A_wins': 0,
            'B_wins': 0,
            'ties': 0,
            'failed': 0,
            'total': len(pairs),
            'details': []
        }
        
        for i, (item_a, item_b) in enumerate(pairs, 1):
            
            result = self.evaluate_pair(item_a, item_b)
            
            if result == 'A':
                results['A_wins'] += 1
            elif result == 'B':
                results['B_wins'] += 1
            elif result == 'tie':
                results['ties'] += 1
            else:
                results['failed'] += 1
            
            results['details'].append({
                'id': i,
                'result': result
            })
        
        total = results['total']
        results['A_win_rate'] = results['A_wins'] / total * 100
        results['B_win_rate'] = results['B_wins'] / total * 100
        results['tie_rate'] = results['ties'] / total * 100
        
        print("\n=== Result ===")
        print(f"Dataset A:B = {results['A_wins']}:{results['B_wins']}")
        print(f"Tie: {results['ties']} ({results['tie_rate']:.1f}%)")
        print(f"Fail: {results['failed']} ({results['failed'] / total * 100:.1f}%)")

        #if output_path:
        #    with open(output_path, 'w', encoding='utf-8') as f:
        #        json.dump(results, f, ensure_ascii=False, indent=2)
        #    print(f"詳細結果已保存至: {output_path}")
        
        return results


def main():
    random.seed(42)
    
    # {"context", "answer", "PAQ", "llama3.2", "qwen3"}
    dataset = "data_example/sampling_results.jsonl"
    output_path = "evaluation_results.json"
    #model_list= ["llama3:70b", "llama3.2:3b", "llama3.2:1b", "qwen3:0.6b", "qwen3:1.7b", "gemma3", "deepseek-r1:8b"]
    model_list = ["llama3.2"]
    results = []
    compare_list = [
        ("PAQ", "llama3.2"),
        ("qwen3", "llama3.2"),
        ("qwen3", "PAQ"),
    ]
    # test each model before evaluating the dataset
    #for model in model_list:
    #    test_llm = OllamaChat(model_name=model)
    #    try:
    #        response, info = test_llm(prompt="Hello, how are you?")
    #        print(f"output from {model}: {response}")
    #    except Exception as e:
    #        print(f"Error with model {model}: {e}")
    #        continue

    for model in model_list:
        print(f"\n=== Direct Scoring: {model} ===")
        direct_evaluator = DatasetScorer(model=model)
        results.append(direct_evaluator.score_dataset(
            dataset_path=dataset,
            output_path=output_path,
        ))
        print(f"\n=== Pairwise comparision: {model} ===")
        pairwise_evaluator = QuestionEvaluator(model=model)
        for dataset_a, dataset_b in compare_list:
            print(f"Comparing {dataset_a} vs {dataset_b}...")
            results.append(pairwise_evaluator.run_evaluation(
                dataset_path=dataset,
                dataset_name_a=dataset_a,
                dataset_name_b=dataset_b,
                output_path=output_path,
            ))

if __name__ == "__main__":
    main()