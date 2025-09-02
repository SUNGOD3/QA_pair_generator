#base.py
from typing import Dict, List, Tuple, Any
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from copy import deepcopy
import json
import pickle
from pathlib import Path

class QAPair:
    
    def __init__(self, context: str = None, question: str = None, answer: str = None, 
                 metadata: Dict[str, Any] = None):
        self.id = None
        self.set_answer(answer)
        self.set_context(context)
        self.set_question(question)
        self.edges = []
        self.metadata = metadata or {}

    def extract_keywords(self, text):
        '''
        Use NLTK to extract the top 5 most common keywords from the text.
        '''
        if not text:
            return []
        words = word_tokenize(text)
        words = [w.lower() for w in words if w.isalnum()] # Remove punctuation
        stop_words = set(stopwords.words("english"))
        filtered_words = [w for w in words if w not in stop_words]
        keyword_counts = Counter(filtered_words)
        return [word for word, _ in keyword_counts.most_common(5)]
    
    def set_context(self, context):
        self.context = context
        self.context_keywords = self.extract_keywords(context)
    
    def set_question(self, question):
        self.question = question
        self.question_keywords = self.extract_keywords(question)
    
    def set_answer(self, answer):
        self.answer = answer
        self.answer_keywords = self.extract_keywords(answer)

    def add_edge(self, target_id, method_name="default"):
        """
        Add an edge to the target ID with a specified method name.
        
        Args:
            target_id: The ID of the target QAPair
            method_name: A string indicating the method used for edge creation
        """
        edge = (method_name, target_id)
        if edge not in self.edges:
            self.edges.append(edge)

    def get_edges_by_method(self, method_name=None):
        """
        Get all edges or edges with a specific method name.
        
        Args:
            method_name: If provided, returns only edges with this method name
            
        Returns:
            A list of target IDs or (method_name, target_id) tuples
        """
        if method_name is None:
            return self.edges
        return [target_id for method, target_id in self.edges if method == method_name]

    def set_metadata(self, key: str, value: Any, method_name: str = None):
        """
        Set metadata for this QAPair.
        
        Args:
            key: The metadata key
            value: The metadata value
            method_name: Optional method name to namespace the metadata
        """
        if method_name:
            if method_name not in self.metadata:
                self.metadata[method_name] = {}
            self.metadata[method_name][key] = value
        else:
            self.metadata[key] = value

    def get_metadata(self, key: str, method_name: str = None, default=None):
        """
        Get metadata for this QAPair.
        """
        if method_name:
            return self.metadata.get(method_name, {}).get(key, default)
        return self.metadata.get(key, default)

    def classify(self):
        '''
        Classify the QA pair based on the presence of context, question, and answer.
        '''
        has_context = self.context is not None
        has_question = self.question is not None
        has_answer = self.answer is not None

        if has_context and not has_question and not has_answer:  # 1 0 0 (1)
            return "Document"
        elif not has_context and has_question and not has_answer:  # 0 1 0 (2)
            return "Question"
        elif not has_context and not has_question and has_answer:  # 0 0 1 (4)
            return "Ground Truth"
        elif has_context and has_question and not has_answer:  # 1 1 0 (3)
            return "Context + Question"
        elif has_context and not has_question and has_answer:  # 1 0 1 (5)
            return "Context + Ground Truth"
        elif not has_context and has_question and has_answer:  # 0 1 1 (6)
            return "QA Pair"
        elif has_context and has_question and has_answer:  # 1 1 1 (7)
            return "Complete QA"
        else:
            return "Unknown"

    def classify_id(self):
        return {
            "Document": 1,
            "Question": 2,
            "Ground Truth": 4,
            "Context + Question": 3,
            "Context + Ground Truth": 5,
            "QA Pair": 6,
            "Complete QA": 7,
            "Unknown": -1
        }[self.classify()]

    def __repr__(self):
        return (f"QAPair(id={self.id}, type={self.classify_id()}, context={self.context}, "
                f"question={self.question}, answer={self.answer}, edges={self.edges}, "
                f"context_keywords={self.context_keywords}, "
                f"question_keywords={self.question_keywords}, "
                f"answer_keywords={self.answer_keywords})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert QAPair to dictionary for serialization."""
        return {
            'id': self.id,
            'context': self.context,
            'question': self.question,
            'answer': self.answer,
            'edges': self.edges,
            'metadata': self.metadata,
            'context_keywords': getattr(self, 'context_keywords', []),
            'question_keywords': getattr(self, 'question_keywords', []),
            'answer_keywords': getattr(self, 'answer_keywords', [])
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QAPair':
        """Create QAPair from dictionary."""
        qa_pair = cls(
            context=data.get('context'),
            question=data.get('question'),
            answer=data.get('answer'),
            metadata=data.get('metadata', {})
        )
        qa_pair.id = data.get('id')
        qa_pair.edges = data.get('edges', [])
        qa_pair.context_keywords = data.get('context_keywords', [])
        qa_pair.question_keywords = data.get('question_keywords', [])
        qa_pair.answer_keywords = data.get('answer_keywords', [])
        return qa_pair


class QADataset:
    def __init__(self, name: str = None, description: str = None, qa_pairs: list[QAPair] = None, 
                 use_disk: bool = True, disk_path: str = None):
        self.name = name or "default_dataset"
        self.description = description
        self.use_disk = use_disk
        self.data: dict[int, QAPair] = {}
        
        # Setup disk storage if enabled
        if self.use_disk:
            self.disk_path = disk_path or f"./qa_dataset_storage/{self.name}"
            self._setup_disk_storage()
        else:
            self.disk_path = None

        if qa_pairs:
            self._initialize_dataset(qa_pairs)

    def _setup_disk_storage(self):
        """Setup disk storage directory and metadata."""
        self.disk_path = Path(self.disk_path)
        self.disk_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        self.metadata_file = self.disk_path / "metadata.json"
        self.data_file = self.disk_path / "qa_data.pkl"
        
        # Initialize metadata if it doesn't exist
        if not self.metadata_file.exists():
            metadata = {
                "name": self.name,
                "description": self.description,
                "size": 0,
                "last_updated": None,
                "current_stage": "init"
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

    def _save_to_disk(self, stage: str = None):
        """Save current dataset to disk."""
        if not self.use_disk:
            return
            
        # Save data
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.data, f)
        
        # Update metadata
        metadata = {
            "name": self.name,
            "description": self.description,
            "size": len(self.data),
            "last_updated": __import__('datetime').datetime.now().isoformat(),
            "current_stage": stage or "unknown"
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to disk at stage: {stage}, size: {len(self.data)}")

    def _load_from_disk(self):
        """Load dataset from disk."""
        if not self.use_disk or not self.data_file.exists():
            return
        
        # Load data
        with open(self.data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        # Load metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                self.name = metadata.get("name", self.name)
                self.description = metadata.get("description", self.description)
        
        print(f"Dataset loaded from disk, size: {len(self.data)}")

    def save_stage_result(self, stage: str):
        """Save the current state after a pipeline stage."""
        if self.use_disk:
            self._save_to_disk(stage)

    def load_stage_input(self, stage: str):
        """Load the dataset state before a pipeline stage."""
        if self.use_disk:
            self._load_from_disk()

    def _initialize_dataset(self, qa_pairs: list[QAPair]):
        '''
        Reassigns the IDs of the QA pairs in the dataset.
        '''
        id_mapping = {} 
        new_data = {}
        for new_id, qa_pair in enumerate(qa_pairs, start=1):
            id_mapping[qa_pair.id] = new_id
            qa_pair.id = new_id
            new_data[new_id] = qa_pair
        
        # Update edges with the new IDs
        for qa in new_data.values():
            new_edges = []
            for edge in qa.edges:
                if isinstance(edge, tuple) and len(edge) == 2:
                    # Handle the case where edge is already (method_name, target_id)
                    method_name, old_id = edge
                    if old_id in id_mapping:
                        new_edges.append((method_name, id_mapping[old_id]))
                else:
                    # Handle legacy edge format (just ID)
                    old_id = edge
                    if old_id in id_mapping:
                        new_edges.append(("default", id_mapping[old_id]))
            qa.edges = new_edges

        self.data = new_data
        
        # Save to disk if enabled
        if self.use_disk:
            self._save_to_disk("initialized")

    def get(self, id):
        if id in self.data:
            return self.data.get(id)
        return None

    def add(self, qa_pair: QAPair):
        new_id = len(self.data) + 1
        if new_id in self.data:
            self.edit(qa_pair.id, qa_pair)
            return
            #raise ValueError("ID already exists in the dataset.")
        new_qa_pair = deepcopy(qa_pair)  
        new_qa_pair.id = new_id
        self.data[new_id] = new_qa_pair

    def add_edge(self, source_id, target_id, method_name="default"):
        """
        Add an edge from source to target with the specified method name.
        
        Args:
            source_id: ID of the source QAPair
            target_id: ID of the target QAPair
            method_name: String indicating the method used for edge creation
        """
        if source_id in self.data and target_id in self.data:
            self.data[source_id].add_edge(target_id, method_name)

    def filter_by_type(self, qa_type: str):
        return [qa for qa in self.data.values() if qa.classify() == qa_type]

    def get_statistics(self):
        '''
        Return the count of each type of QA pair in the dataset.
        '''
        stats = {}
        for qa in self.data.values():
            qa_type = qa.classify()
            stats[qa_type] = stats.get(qa_type, 0) + 1
        return stats
    
    def delete(self, ids_to_delete: list[int]):
        """
        Delete QAPairs with specified IDs and maintain data integrity.
        
        Args:
            ids_to_delete: List of IDs to delete from the dataset
        """
        if not ids_to_delete:
            return
        
        # Convert to set for O(1) lookups
        ids_to_delete_set = set(ids_to_delete)
        
        # Remove entries with the specified IDs
        remaining_data = {id: qa_pair for id, qa_pair in self.data.items() 
                        if id not in ids_to_delete_set}
        
        # Remove edges pointing to deleted QAPairs
        for qa_pair in remaining_data.values():
            qa_pair.edges = [(method, target_id) for method, target_id in qa_pair.edges 
                            if target_id not in ids_to_delete_set]
        
        # reassign IDs and update edges
        remaining_qa_pairs = list(remaining_data.values())
        self._initialize_dataset(remaining_qa_pairs)

    def edit(self, id: int, qa_pair: QAPair = None, context: str = None, question: str = None, answer: str = None):
        """
        Edit an existing QAPair in the dataset by ID.
        
        Args:
            id: The ID of the QAPair to edit
            qa_pair: A QAPair object to replace the existing one (optional)
            context: New context text (optional if qa_pair is provided)
            question: New question text (optional if qa_pair is provided)
            answer: New answer text (optional if qa_pair is provided)
            
        Returns:
            bool: True if the edit was successful, False otherwise
        
        Note:
            If qa_pair is provided, it will override any individual field parameters.
            The ID of the provided qa_pair will be ignored and the original ID maintained.
        """
        if id not in self.data:
            return False
            
        if qa_pair:
            # Create a deep copy to avoid modifying the input qa_pair
            new_qa_pair = deepcopy(qa_pair)
            # Preserve the original ID and edges
            original_id = self.data[id].id
            original_edges = self.data[id].edges
            new_qa_pair.id = original_id
            new_qa_pair.edges = original_edges
            self.data[id] = new_qa_pair
        elif context is not None or question is not None or answer is not None:
            # Update individual fields if provided
            if context is not None:
                self.data[id].set_context(context)
            if question is not None:
                self.data[id].set_question(question)
            if answer is not None:
                self.data[id].set_answer(answer)
        else:
            raise ValueError("No valid fields provided to edit.")
        print(f"Edited QAPair with ID {id}: {self.data[id]}")
        return True

    def __iter__(self):
        return iter(self.data.values())

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        disk_info = f", disk_enabled={self.use_disk}" if self.use_disk else ""
        return f"QADataset(size={len(self.data)}{disk_info})"
    
    def copy(self):
        return QADataset(
            name=self.name,
            description=self.description,
            qa_pairs=list(self.data.values()),
            use_disk=self.use_disk,
            disk_path=self.disk_path
        )