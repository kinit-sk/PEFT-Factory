# Copyright 2025 the PEFTFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import importlib.util
from pathlib import Path
from typing import Type, Dict, Tuple, Optional
from ..extras import logging

from peft import PeftConfig
from peft.tuners.tuners_utils import BaseTuner

logger = logging.get_logger(__name__)


def discover_custom_peft_methods(peft_dir: str) -> Dict[str, Tuple[Type[PeftConfig], Type[BaseTuner]]]:
    """
    Discover custom PEFT methods from directory structure.
    
    Args:
        peft_dir: Path to the peft directory containing custom PEFT implementations
        
    Returns:
        Dictionary mapping method names to (ConfigClass, ModelClass) tuples
    """
    discovered_methods = {}
    peft_path = Path(peft_dir)
    
    if not peft_path.exists():
        logger.warning_rank0(f"PEFT directory not found: {peft_dir}")
        return discovered_methods
    
    if not peft_path.is_dir():
        logger.warning_rank0(f"PEFT path is not a directory: {peft_dir}")
        return discovered_methods
    
    # Scan each subdirectory in the peft directory
    try:
        for method_dir in peft_path.iterdir():
            if not method_dir.is_dir():
                continue
                
            method_name = method_dir.name
            
            # Skip hidden directories and __pycache__
            if method_name.startswith('.') or method_name == '__pycache__':
                continue
                
            try:
                config_cls, model_cls = _load_peft_method(method_dir, method_name)
                if config_cls and model_cls:
                    discovered_methods[method_name] = (config_cls, model_cls)
                    logger.info_rank0(f"Discovered custom PEFT method: {method_name}")
                else:
                    logger.warning_rank0(f"Failed to load complete PEFT method '{method_name}': missing config or model class")
            except Exception as e:
                logger.warning_rank0(f"Failed to load PEFT method '{method_name}': {e}")
                continue
    except PermissionError as e:
        logger.error(f"Permission denied accessing PEFT directory {peft_dir}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error scanning PEFT directory {peft_dir}: {e}")
    
    return discovered_methods


def _load_peft_method(method_dir: Path, method_name: str) -> Tuple[Optional[Type[PeftConfig]], Optional[Type[BaseTuner]]]:
    """
    Load config and model classes for a specific PEFT method.
    
    Args:
        method_dir: Directory containing the PEFT method implementation
        method_name: Name of the PEFT method
        
    Returns:
        Tuple of (ConfigClass, ModelClass) or (None, None) if loading fails
    """
    config_cls = None
    model_cls = None
    
    # Load config.py
    config_file = method_dir / "config.py"
    if config_file.exists():
        try:
            config_module = _import_module_from_path(config_file, f"{method_name}_config")
            config_cls = _find_peft_config_class(config_module, method_name)
        except Exception as e:
            logger.warning_rank0(f"Failed to load config for {method_name}: {e}")
    else:
        logger.warning_rank0(f"No config.py found in {method_dir}")
        return None, None
    
    # Load model.py
    model_file = method_dir / "model.py"
    if model_file.exists():
        try:
            model_module = _import_module_from_path(model_file, f"{method_name}_model")
            model_cls = _find_base_tuner_class(model_module, method_name)
        except Exception as e:
            logger.warning_rank0(f"Failed to load model for {method_name}: {e}")
    else:
        logger.warning_rank0(f"No model.py found in {method_dir}")
        return None, None
    
    return config_cls, model_cls


def _import_module_from_path(file_path: Path, module_name: str):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _find_peft_config_class(module, method_name: str) -> Optional[Type[PeftConfig]]:
    """Find a class that inherits from PeftConfig in the module."""
    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, PeftConfig) and 
                attr != PeftConfig):
                # Validate that the class has required attributes
                if hasattr(attr, 'peft_type'):
                    return attr
                else:
                    logger.warning_rank0(f"Config class {attr_name} in {method_name} missing 'peft_type' attribute")
        except (TypeError, AttributeError) as e:
            # Skip attributes that can't be checked for inheritance
            continue
    
    logger.warning_rank0(f"No valid PeftConfig subclass found in {method_name} config module")
    return None


def _find_base_tuner_class(module, method_name: str) -> Optional[Type[BaseTuner]]:
    """Find a class that inherits from BaseTuner in the module."""
    for attr_name in dir(module):
        try:
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BaseTuner) and 
                attr != BaseTuner):
                # Validate that the class has required attributes
                if hasattr(attr, 'prefix'):
                    return attr
                else:
                    logger.warning_rank0(f"Model class {attr_name} in {method_name} missing 'prefix' attribute")
        except (TypeError, AttributeError) as e:
            # Skip attributes that can't be checked for inheritance
            continue
    
    logger.warning_rank0(f"No valid BaseTuner subclass found in {method_name} model module")
    return None


def get_custom_peft_config(method_name: str) -> Optional[Type[PeftConfig]]:
    """
    Get config class for a custom PEFT method.
    
    Args:
        method_name: Name of the PEFT method
        
    Returns:
        Config class or None if not found
    """
    # This would be used if we cache the discovered methods
    # For now, we'll rely on the constants module to handle this
    return None


def get_custom_peft_model(method_name: str) -> Optional[Type[BaseTuner]]:
    """
    Get model class for a custom PEFT method.
    
    Args:
        method_name: Name of the PEFT method
        
    Returns:
        Model class or None if not found
    """
    # This would be used if we cache the discovered methods
    # For now, we'll rely on the constants module to handle this
    return None
