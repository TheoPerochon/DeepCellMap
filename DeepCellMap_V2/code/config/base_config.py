import json
import os
import inspect 
from copy import deepcopy 
from typing import Any, Dict


class BaseConfig:
    def __init__(self, dataset_name):
        """
        Initialize with a dataset name and set up paths based on that name.
        """
        self.dataset_name = dataset_name
        self.debug_mode = False
        self.update_paths()

    def get_base_path(self):
        """
        Calculate the base path dynamically based on the script's location.
        This can be modified as per your directory structure.
        """
        return os.path.dirname(os.path.abspath(__file__))

    def update_paths(self):
        """
        Update all paths that depend on dataset_name.
        """
        self.base_path = self.get_base_path()
        self.data_dir = os.path.join(os.path.dirname(self.base_path), "data")
        self.config_dir = os.path.join(self.base_path, 'config')
        self.dir_output = os.path.join(os.path.dirname(self.base_path), 'outputs')
        self.dir_dataset = os.path.join(self.data_dir, self.dataset_name)
        self.dir_output_dataset = os.path.join(self.dir_output, self.dataset_name)

    def print_config(self):
        """
        Print the configuration.
        """
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

    def get_base_path(self):
        """
        Get the base path of the currently executing script.
        """
        # Get the directory containing the script being executed
        config_dir = os.path.dirname(os.path.abspath(__file__))
        code_path = os.path.dirname(config_dir)
        base_path = os.path.dirname(code_path)
        return base_path
    
    def save(self, config_file):
        """
        Save the configuration to a JSON file.

        Args:
            config_file (str): The path to the JSON file where the configuration will be saved.
        """
        try:
            with open(config_file, 'w') as json_file:
                json.dump(self.__dict__, json_file, indent=4)
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")

    def load(self, config_file):
        """
        Load the configuration from a JSON file.

        Args:
            config_file (str): The path to the JSON file containing the configuration.
        """
        try:
            with open(config_file, 'r') as json_file:
                loaded_config = json.load(json_file)
                # Update the configuration object's attributes with the loaded data
                self.__dict__.update(loaded_config)
        except FileNotFoundError:
            print(f"Configuration file '{config_file}' not found.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in '{config_file}': {str(e)}")
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")


    def to_dict(self) -> Dict[str, Any]:
        """Export config to a python dict."""
        # List all public attributes, excluding methods and functions 
        attributes = [ 
            name for name in dir(self) 
            if not name.startswith(" ") and not inspect.ismethod(getattr(self, name))
            ]
        return deepcopy({name: getattr(self, name) for name in attributes})


    def to_json(self, filename: str):
        """Export config to disk in json format."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, sort_keys=True)

    @classmethod
    def from_dict(cls, config_dict):
        """Create a config object from a python dict."""
        config = cls()
        config_dict = deepcopy(config_dict)
        for key, value in config_dict.items(): 
            setattr(config, key, value)
        return config
    
    @classmethod
    def from_json(cls, filename):
        """Create a config object from a json file."""
        with open(filename, "r", encoding="utf-8") as f: 
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def clone(self):
        """Clone config. """
        return self.__class__.from_dict(self.to_dict())