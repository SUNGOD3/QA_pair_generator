# method_register.py
import os
import importlib
import inspect
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path


class Method:
    """
    Decorator class to automatically register methods with metadata.
    Stores all discovered methods with their associated information.
    """
    # Class-level dictionary to store all discovered methods
    _methods: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self, 
        name: str, 
        description: str, 
        applicable_stages: List[str], # List[str] = ["init_setup", "data_expansion", "build_knowledge_graph", "data_fusion", "data_filter", "data_augmentation"]
        use_LLM: bool = False,
        use_docker: bool = False,
    ):
        """
        Initialize a method with its metadata.
        
        Args:
            name (str): Unique name of the method
            description (str): Detailed description of the method
            applicable_stages (List[str]): Stages where this method can be applied
            use_LLM (bool, optional): Whether the method uses LLM. Defaults to False.
        """
        self.name = name
        self.description = description
        self.applicable_stages = applicable_stages
        self.use_LLM = use_LLM
        self.use_docker = use_docker

    def __call__(self, func: Callable):
        """
        Register the method when the decorator is applied.
        
        Args:
            func (Callable): The method function to be registered
        
        Returns:
            Callable: The original function
        """
        # Store method information
        method_info = {
            'func': func,
            'name': self.name,
            'description': self.description,
            'applicable_stages': self.applicable_stages,
            'use_LLM': self.use_LLM,
            'use_docker': self.use_docker
        }
            
        Method._methods[self.name] = method_info
        return func

    @classmethod
    def get_methods(cls) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve all registered methods.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of registered methods
        """
        return cls._methods

    @classmethod
    def get_methods_for_stage(cls, stage: str) -> Dict[str, Dict[str, Any]]:
        """
        Get methods applicable to a specific stage.
        
        Args:
            stage (str): Pipeline stage to filter methods for
        
        Returns:
            Dict[str, Dict[str, Any]]: Methods applicable to the stage
        """
        return {
            name: method for name, method in cls._methods.items()
            if stage in method['applicable_stages']
        }

    @classmethod
    def clear_methods(cls):
        """Clear all registered methods. Useful for testing."""
        cls._methods.clear()


class MethodRegistry:
    """
    Registry class to automatically discover and load methods from the methods directory.
    """
    
    def __init__(self, methods_dir: str = "methods"):
        """
        Initialize the method registry.
        
        Args:
            methods_dir (str): Directory containing method files
        """
        self.methods_dir = methods_dir
        self.loaded_modules = {}
        
    def discover_and_load_methods(self, base_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Discover and load all methods from the methods directory.
        
        Args:
            base_path (Optional[str]): Base path to look for methods directory.
                                     If None, uses current directory.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of all loaded methods
        """
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            
        methods_path = Path(base_path) / self.methods_dir
        
        if not methods_path.exists():
            print(f"Warning: Methods directory '{methods_path}' does not exist")
            return {}
            
        # Clear existing methods to avoid duplicates
        Method.clear_methods()
        
        # Get all Python files in the methods directory
        method_files = list(methods_path.glob("*.py"))
        method_files = [f for f in method_files if f.name != "__init__.py"]
        
        print(f"Discovering methods in {methods_path}...")
        print(f"Found {len(method_files)} method files: {[f.name for f in method_files]}")
        
        # Import each method file
        for method_file in method_files:
            try:
                self._load_method_file(method_file, methods_path)
            except Exception as e:
                print(f"Error loading method file {method_file.name}: {e}")
                continue
        
        loaded_methods = Method.get_methods()
        print(f"Successfully loaded {len(loaded_methods)} methods: {list(loaded_methods.keys())}")
        
        return loaded_methods
    
    def _load_method_file(self, method_file: Path, methods_path: Path):
        """
        Load a single method file.
        
        Args:
            method_file (Path): Path to the method file
            methods_path (Path): Path to the methods directory
        """
        module_name = method_file.stem
        
        # Create module spec and load the module
        spec = importlib.util.spec_from_file_location(
            f"methods.{module_name}", 
            method_file
        )
        
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {method_file}")
        
        module = importlib.util.module_from_spec(spec)
        
        # Store reference to avoid garbage collection
        self.loaded_modules[module_name] = module
        
        # Execute the module to trigger method registration
        spec.loader.exec_module(module)
        
        print(f"  Loaded method file: {method_file.name}")
    
    def reload_methods(self, base_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Reload all methods (useful for development).
        
        Args:
            base_path (Optional[str]): Base path to look for methods directory
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of all reloaded methods
        """
        # Clear loaded modules
        self.loaded_modules.clear()
        
        # Reload methods
        return self.discover_and_load_methods(base_path)
    
    def get_method_info(self, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific method.
        
        Args:
            method_name (str): Name of the method
        
        Returns:
            Optional[Dict[str, Any]]: Method information or None if not found
        """
        methods = Method.get_methods()
        return methods.get(method_name)
    
    def list_methods_by_stage(self, stage: str) -> List[str]:
        """
        List all method names applicable to a specific stage.
        
        Args:
            stage (str): Pipeline stage
        
        Returns:
            List[str]: List of method names
        """
        stage_methods = Method.get_methods_for_stage(stage)
        return list(stage_methods.keys())
    
    def validate_methods(self) -> Dict[str, List[str]]:
        """
        Validate all loaded methods for common issues.
        
        Returns:
            Dict[str, List[str]]: Dictionary of validation results
        """
        results = {
            "valid": [],
            "warnings": [],
            "errors": []
        }
        
        methods = Method.get_methods()
        
        for name, method_info in methods.items():
            try:
                # Check if function is callable
                if not callable(method_info['func']):
                    results["errors"].append(f"{name}: Function is not callable")
                    continue
                
                # Check function signature
                sig = inspect.signature(method_info['func'])
                params = list(sig.parameters.keys())
                
                if len(params) < 2:
                    results["warnings"].append(f"{name}: Function should accept at least 2 parameters (qa_pairs, config)")
                
                # Check if required fields are present
                required_fields = ['name', 'description', 'applicable_stages', 'use_LLM', 'use_docker']
                missing_fields = [field for field in required_fields if field not in method_info]
                
                if missing_fields:
                    results["errors"].append(f"{name}: Missing required fields: {missing_fields}")
                    continue
                
                # Check applicable_stages format
                if not isinstance(method_info['applicable_stages'], list):
                    results["errors"].append(f"{name}: applicable_stages must be a list")
                    continue
                
                results["valid"].append(name)
                
            except Exception as e:
                results["errors"].append(f"{name}: Validation error: {str(e)}")
        
        return results


# Global registry instance
method_registry = MethodRegistry()


def load_all_methods(base_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to load all methods.
    
    Args:
        base_path (Optional[str]): Base path to look for methods directory
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of all loaded methods
    """
    return method_registry.discover_and_load_methods(base_path)


def get_methods_for_stage(stage: str) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to get methods for a specific stage.
    
    Args:
        stage (str): Pipeline stage
    
    Returns:
        Dict[str, Dict[str, Any]]: Methods applicable to the stage
    """
    return Method.get_methods_for_stage(stage)