"""Dataset loading and management utilities."""

import json
import random
from typing import Dict, Any, List, Optional, Iterator
from pathlib import Path
import logging
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Base class for datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"dataset.{self.__class__.__name__}")
        
    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Load the dataset."""
        pass
    
    @abstractmethod
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a specific sample."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get dataset length."""
        pass


class MMLUDataset(BaseDataset):
    """MMLU dataset loader."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.subjects = config.get("subjects", ["mathematics", "physics", "chemistry", "biology", "history"])
        self.samples_per_subject = config.get("samples_per_subject", 20)
        self.data = []
        
    def load(self) -> List[Dict[str, Any]]:
        """Load MMLU samples."""
        if self.data:
            return self.data
        
        # Check if dataset file exists
        data_path = Path(self.config.get("data_path", "evals/mmlu_subset.jsonl"))
        
        if data_path.exists():
            self.data = self._load_from_file(data_path)
        else:
            self.data = self._generate_mock_data()
            
        self.logger.info(f"Loaded {len(self.data)} MMLU samples")
        return self.data
    
    def _load_from_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load from JSONL file."""
        samples = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        return samples
    
    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock MMLU data."""
        samples = []
        
        for subject in self.subjects:
            for i in range(self.samples_per_subject):
                sample = {
                    "id": f"mmlu_{subject}_{i}",
                    "subject": subject,
                    "question": f"Sample {subject} question {i}: What is the primary concept in {subject}?",
                    "choices": [
                        f"A) {subject} concept A",
                        f"B) {subject} concept B", 
                        f"C) {subject} concept C",
                        f"D) {subject} concept D"
                    ],
                    "answer": random.choice(["A", "B", "C", "D"]),
                    "difficulty": random.choice(["easy", "medium", "hard"])
                }
                samples.append(sample)
        
        return samples
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get sample by index."""
        if not self.data:
            self.load()
        return self.data[index]
    
    def __len__(self) -> int:
        """Get dataset length."""
        if not self.data:
            self.load()
        return len(self.data)
    
    def get_by_subject(self, subject: str) -> List[Dict[str, Any]]:
        """Get samples for a specific subject."""
        if not self.data:
            self.load()
        return [sample for sample in self.data if sample.get("subject") == subject]


class GSM8KDataset(BaseDataset):
    """GSM8K dataset loader."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_samples = config.get("num_samples", 100)
        self.data = []
        
    def load(self) -> List[Dict[str, Any]]:
        """Load GSM8K samples."""
        if self.data:
            return self.data
        
        data_path = Path(self.config.get("data_path", "evals/gsm8k_subset.jsonl"))
        
        if data_path.exists():
            self.data = self._load_from_file(data_path)
        else:
            self.data = self._generate_mock_data()
            
        self.logger.info(f"Loaded {len(self.data)} GSM8K samples")
        return self.data
    
    def _load_from_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load from JSONL file."""
        samples = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        return samples
    
    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock GSM8K data."""
        samples = []
        
        problem_templates = [
            "John has {a} apples. He gives {b} apples to his friend. How many apples does John have left?",
            "Sarah bought {a} pencils for ${b} each. How much did she spend in total?",
            "A store has {a} items. If {b}% are sold, how many items are left?",
            "Mike runs {a} miles per day for {b} days. How many total miles did he run?"
        ]
        
        for i in range(self.num_samples):
            template = random.choice(problem_templates)
            a = random.randint(10, 100)
            b = random.randint(5, 30)
            
            question = template.format(a=a, b=b)
            
            # Calculate answer based on template
            if "gives" in template:
                answer = a - b
            elif "bought" in template:
                answer = a * b
            elif "%" in template:
                answer = a * (1 - b/100)
            elif "runs" in template:
                answer = a * b
            else:
                answer = a + b
            
            sample = {
                "id": f"gsm8k_{i}",
                "question": question,
                "answer": int(answer),
                "solution_steps": [
                    "Step 1: Identify the given information",
                    "Step 2: Determine the operation needed",
                    "Step 3: Perform the calculation", 
                    f"Step 4: The answer is {answer}"
                ]
            }
            samples.append(sample)
        
        return samples
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get sample by index."""
        if not self.data:
            self.load()
        return self.data[index]
    
    def __len__(self) -> int:
        """Get dataset length."""
        if not self.data:
            self.load()
        return len(self.data)


class MTBenchDataset(BaseDataset):
    """MT-Bench dataset loader."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.categories = config.get("categories", ["writing", "roleplay", "reasoning", "math", "coding"])
        self.samples_per_category = config.get("samples_per_category", 16)
        self.data = []
        
    def load(self) -> List[Dict[str, Any]]:
        """Load MT-Bench samples."""
        if self.data:
            return self.data
        
        data_path = Path(self.config.get("data_path", "evals/mtbench_subset.jsonl"))
        
        if data_path.exists():
            self.data = self._load_from_file(data_path)
        else:
            self.data = self._generate_mock_data()
            
        self.logger.info(f"Loaded {len(self.data)} MT-Bench samples")
        return self.data
    
    def _load_from_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load from JSONL file."""
        samples = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        return samples
    
    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock MT-Bench data."""
        samples = []
        
        prompt_templates = {
            "writing": [
                "Write a creative story about {topic}",
                "Compose a professional email about {topic}",
                "Write a persuasive essay on {topic}"
            ],
            "roleplay": [
                "Act as a {role} and help with {topic}",
                "Pretend you are a {role} explaining {topic}",
                "Role-play as a {role} discussing {topic}"
            ],
            "reasoning": [
                "Analyze the logical structure of {topic}",
                "Explain the cause and effect relationship in {topic}",
                "Break down the reasoning behind {topic}"
            ],
            "math": [
                "Solve this mathematical problem: {topic}",
                "Explain the mathematical concept of {topic}",
                "Calculate and show steps for {topic}"
            ],
            "coding": [
                "Write a Python function to {topic}",
                "Debug this code snippet for {topic}",
                "Implement an algorithm for {topic}"
            ]
        }
        
        topics = ["data analysis", "machine learning", "web development", "optimization", 
                 "problem solving", "automation", "visualization", "algorithms"]
        roles = ["teacher", "consultant", "expert", "advisor", "specialist"]
        
        for category in self.categories:
            templates = prompt_templates.get(category, ["Discuss {topic}"])
            
            for i in range(self.samples_per_category):
                template = random.choice(templates)
                topic = random.choice(topics)
                role = random.choice(roles) if "{role}" in template else None
                
                if role:
                    prompt = template.format(role=role, topic=topic)
                else:
                    prompt = template.format(topic=topic)
                
                sample = {
                    "id": f"mtbench_{category}_{i}",
                    "category": category,
                    "conversation": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "reference_answer": f"This is a reference answer for {category} task about {topic}.",
                    "scoring_criteria": ["relevance", "coherence", "accuracy", "helpfulness"]
                }
                samples.append(sample)
        
        return samples
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get sample by index."""
        if not self.data:
            self.load()
        return self.data[index]
    
    def __len__(self) -> int:
        """Get dataset length."""
        if not self.data:
            self.load()
        return len(self.data)
    
    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get samples for a specific category."""
        if not self.data:
            self.load()
        return [sample for sample in self.data if sample.get("category") == category]


class CalibrationDataset(BaseDataset):
    """Calibration dataset for quantization."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_samples = config.get("num_samples", 512)
        self.sequence_length = config.get("sequence_length", 512)
        self.data = []
        
    def load(self) -> List[Dict[str, Any]]:
        """Load calibration samples."""
        if self.data:
            return self.data
        
        data_path = Path(self.config.get("data_path", "evals/calibration.jsonl"))
        
        if data_path.exists():
            self.data = self._load_from_file(data_path)
        else:
            self.data = self._generate_mock_data()
            
        self.logger.info(f"Loaded {len(self.data)} calibration samples")
        return self.data
    
    def _load_from_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load from JSONL file."""
        samples = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        return samples
    
    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock calibration data."""
        samples = []
        
        # Generate diverse text samples
        domains = ["technology", "science", "literature", "history", "business"]
        
        for i in range(self.num_samples):
            domain = random.choice(domains)
            
            # Generate text of appropriate length
            words = []
            target_words = self.sequence_length // 4  # Rough token estimation
            
            for j in range(target_words):
                if j % 10 == 0:  # Add domain-specific terms periodically
                    words.append(f"{domain}_term_{j}")
                else:
                    words.append(f"word_{j}")
            
            text = " ".join(words)
            
            sample = {
                "id": f"calib_{i}",
                "text": text,
                "domain": domain,
                "length": len(words),
                "source": "synthetic"
            }
            samples.append(sample)
        
        return samples
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get sample by index."""
        if not self.data:
            self.load()
        return self.data[index]
    
    def __len__(self) -> int:
        """Get dataset length."""
        if not self.data:
            self.load()
        return len(self.data)
    
    def get_texts(self) -> List[str]:
        """Get just the text content for calibration."""
        if not self.data:
            self.load()
        return [sample["text"] for sample in self.data]


class DatasetFactory:
    """Factory for creating datasets."""
    
    @staticmethod
    def create_dataset(dataset_type: str, config: Dict[str, Any]) -> BaseDataset:
        """Create a dataset of the specified type."""
        
        if dataset_type.lower() == "mmlu":
            return MMLUDataset(config)
        elif dataset_type.lower() == "gsm8k":
            return GSM8KDataset(config)
        elif dataset_type.lower() == "mtbench":
            return MTBenchDataset(config)
        elif dataset_type.lower() == "calibration":
            return CalibrationDataset(config)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    @staticmethod
    def get_available_datasets() -> List[str]:
        """Get list of available dataset types."""
        return ["mmlu", "gsm8k", "mtbench", "calibration"]


class DatasetManager:
    """Manager for handling multiple datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("dataset_manager")
        self.datasets = {}
        
    def load_dataset(self, name: str, dataset_type: str, 
                    dataset_config: Dict[str, Any]) -> BaseDataset:
        """Load a dataset."""
        
        if name in self.datasets:
            self.logger.info(f"Dataset {name} already loaded")
            return self.datasets[name]
        
        try:
            dataset = DatasetFactory.create_dataset(dataset_type, dataset_config)
            dataset.load()
            self.datasets[name] = dataset
            
            self.logger.info(f"Loaded dataset {name} ({dataset_type}) with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset {name}: {e}")
            raise
    
    def get_dataset(self, name: str) -> Optional[BaseDataset]:
        """Get a loaded dataset."""
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """List all loaded datasets."""
        return list(self.datasets.keys())
    
    def create_mixed_dataset(self, name: str, source_datasets: List[str], 
                           samples_per_dataset: Optional[int] = None) -> BaseDataset:
        """Create a mixed dataset from multiple sources."""
        
        # Collect samples from source datasets
        all_samples = []
        
        for dataset_name in source_datasets:
            if dataset_name not in self.datasets:
                self.logger.warning(f"Dataset {dataset_name} not loaded, skipping")
                continue
            
            dataset = self.datasets[dataset_name]
            samples = dataset.load()
            
            if samples_per_dataset:
                samples = samples[:samples_per_dataset]
            
            # Add source information
            for sample in samples:
                sample["source_dataset"] = dataset_name
            
            all_samples.extend(samples)
        
        # Shuffle the mixed samples
        random.shuffle(all_samples)
        
        # Create a simple dataset wrapper
        class MixedDataset(BaseDataset):
            def __init__(self, samples):
                self.samples = samples
                
            def load(self):
                return self.samples
                
            def get_sample(self, index):
                return self.samples[index]
                
            def __len__(self):
                return len(self.samples)
        
        mixed_dataset = MixedDataset(all_samples)
        self.datasets[name] = mixed_dataset
        
        self.logger.info(f"Created mixed dataset {name} with {len(all_samples)} samples")
        return mixed_dataset