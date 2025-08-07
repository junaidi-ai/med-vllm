import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Adapter configuration
    use_medical_adapter: bool = True
    adapter_type: str | None = None  # Auto-detect if None
    adapter_config: dict | None = None
    use_cuda_graphs: bool = False

    # CUDA optimization settings
    memory_efficient: bool = True
    enable_mixed_precision: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create a Config instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            A new Config instance
        """
        # Filter out None values to use defaults for missing keys
        filtered_dict = {k: v for k, v in config_dict.items() if v is not None}
        return cls(**filtered_dict)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
            if hasattr(self.hf_config, 'max_position_embeddings')
            else self.max_model_len
        )
    
    def __eq__(self, other):
        print("\n[DEBUG] ===== Starting equality comparison =====")
        
        # Quick reference check
        if self is other:
            print("[DEBUG] Same object reference, returning True")
            return True
            
        # Type check
        if not isinstance(other, type(self)):
            print(f"[DEBUG] Not equal: different types: {type(self)} vs {type(other)}")
            return False
        
        # Get all fields from both objects, including those set after initialization
        self_dict = self.__dict__.copy()
        other_dict = other.__dict__.copy()
        
        # Get all field names from dataclass fields and actual instance attributes
        all_fields = set(self_dict.keys()).union(other_dict.keys())
        
        print("\n[DEBUG] ===== Field Comparison =====")
        
        # Special handling for hf_config - compare model_type only
        if hasattr(self, 'hf_config') and hasattr(other, 'hf_config'):
            print("\n[DEBUG] Comparing hf_config...")
            self_hf_type = getattr(self.hf_config, 'model_type', None)
            other_hf_type = getattr(other.hf_config, 'model_type', None)
            
            print(f"  Self hf_config.model_type: {self_hf_type}")
            print(f"  Other hf_config.model_type: {other_hf_type}")
            
            if self_hf_type != other_hf_type:
                print(f"[DEBUG] hf_config.model_type differs: {self_hf_type} != {other_hf_type}")
                return False
            print("  ✓ hf_config.model_type matches")
        
        # Track mismatches for better debugging
        mismatches = []
        
        # Compare all other fields
        for field in sorted(all_fields):
            try:
                # Skip hf_config as we've already handled it
                if field == 'hf_config':
                    print("\n[DEBUG] Skipping hf_config as it was already compared")
                    continue
                    
                self_val = getattr(self, field, None)
                other_val = getattr(other, field, None)
                
                print(f"\n[DEBUG] Comparing field: {field}")
                print(f"  Self val: {self_val} (type: {type(self_val)})")
                print(f"  Other val: {other_val} (type: {type(other_val)})")
                
                # Skip private fields and config_version
                if field.startswith('_') or field == 'config_version':
                    print(f"  {field}: [SKIPPED]")
                    continue
                    
                # Handle None values
                if self_val is None and other_val is None:
                    print("  Both values are None, considering equal")
                    continue
                    
                if self_val is None or other_val is None:
                    print(f"[DEBUG] Not equal: One value is None: {self_val} vs {other_val}")
                    print(f"[DEBUG] Field {field} differs: {self_val} != {other_val}")
                    print(f"[DEBUG] Self object: {self}")
                    print(f"[DEBUG] Other object: {other}")
                    return False
                
                # Special handling for Path objects to compare string representations
                if hasattr(self_val, '__fspath__') or hasattr(other_val, '__fspath__'):
                    print("  Comparing Path-like objects as strings...")
                    self_str = str(self_val) if hasattr(self_val, '__fspath__') else self_val
                    other_str = str(other_val) if hasattr(other_val, '__fspath__') else other_val
                    
                    if isinstance(self_str, str) and isinstance(other_str, str):
                        if os.path.normpath(self_str) != os.path.normpath(other_str):
                            print(f"[DEBUG] Not equal: Path field {field} differs: {self_str} != {other_str}")
                            print(f"[DEBUG] Field {field} differs: {self_val} != {other_val}")
                            print(f"[DEBUG] Self object: {self}")
                            print(f"[DEBUG] Other object: {other}")
                            return False
                        print("  Paths are equal")
                        continue
                    else:
                        print(f"  Warning: Could not compare as paths: {type(self_str)} vs {type(other_str)}")
                
                # Handle enum/string normalization
                if isinstance(self_val, str) or isinstance(other_val, str) or hasattr(self_val, 'value') or hasattr(other_val, 'value'):
                    print("  Handling string/enum comparison...")
                    self_str = str(self_val.value) if hasattr(self_val, 'value') else str(self_val)
                    other_str = str(other_val.value) if hasattr(other_val, 'value') else str(other_val)
                    
                    if isinstance(self_str, str) and isinstance(other_str, str):
                        if self_str.lower() == other_str.lower():
                            print(f"  String/enum values match after normalization: {self_str} == {other_str}")
                            continue
                # Special handling for lists to compare elements
                if isinstance(self_val, list) and isinstance(other_val, list):
                    print("  Comparing lists...")
                    if len(self_val) != len(other_val):
                        print(f"  Lists have different lengths: {len(self_val)} vs {len(other_val)}")
                        return False
                        
                    for i, (self_item, other_item) in enumerate(zip(self_val, other_val)):
                        if isinstance(self_item, (str, int, float, bool)) and isinstance(other_item, (str, int, float, bool)):
                            if str(self_item).lower() != str(other_item).lower():
                                print(f"  List items at index {i} differ: {self_item} != {other_item}")
                                return False
                        elif self_item != other_item:
                            print(f"  List items at index {i} differ: {self_item} != {other_item}")
                            return False
                    print("  Lists are equal")
                    continue
                    
                # Special handling for dictionaries to compare key-value pairs
                if isinstance(self_val, dict) and isinstance(other_val, dict):
                    print("  Comparing dictionaries...")
                    if set(self_val.keys()) != set(other_val.keys()):
                        print(f"  Dictionaries have different keys: {set(self_val.keys())} vs {set(other_val.keys())}")
                        return False
                        
                    for key in self_val:
                        if isinstance(self_val[key], (str, int, float, bool)) and isinstance(other_val[key], (str, int, float, bool)):
                            if str(self_val[key]).lower() != str(other_val[key]).lower():
                                print(f"  Dictionary values for key '{key}' differ: {self_val[key]} != {other_val[key]}")
                                return False
                        elif self_val[key] != other_val[key]:
                            print(f"  Dictionary values for key '{key}' differ: {self_val[key]} != {other_val[key]}")
                            return False
                    print("  Dictionaries are equal")
                    continue
                    
                # Default comparison for other types
                if isinstance(self_val, (str, int, float, bool)) and isinstance(other_val, (str, int, float, bool)):
                    if str(self_val).lower() != str(other_val).lower():
                        print(f"[DEBUG] Not equal: Field {field} differs: {self_val} != {other_val}")
                        return False
                    print(f"  Values match after string conversion: {self_val} == {other_val}")
                    continue
                    
                # Fallback to direct comparison
                if self_val != other_val:
                    print(f"[DEBUG] Not equal: Field {field} differs: {self_val} != {other_val}")
                    print(f"  Type of self.{field}: {type(self_val)}")
                    print(f"  Type of other.{field}: {type(other_val)}")
                    
                    # Print more details for complex objects
                    if hasattr(self_val, '__dict__') and hasattr(other_val, '__dict__'):
                        print(f"  Self {field} dict: {self_val.__dict__}")
                        print(f"  Other {field} dict: {other_val.__dict__}")
                    
                    # Print all fields of the objects being compared
                    print("\nAll fields in self:", sorted(self_dict.keys()))
                    print("All fields in other:", sorted(other_dict.keys()))
                    
                    # Print any fields that exist in one but not the other
                    only_in_self = set(self_dict.keys()) - set(other_dict.keys())
                    only_in_other = set(other_dict.keys()) - set(self_dict.keys())
                    
                    if only_in_self:
                        print("\nFields only in self:")
                        for f in sorted(only_in_self):
                            print(f"  {f}: {getattr(self, f, 'N/A')}")
                    
                    if only_in_other:
                        print("\nFields only in other:")
                        for f in sorted(only_in_other):
                            print(f"  {f}: {getattr(other, f, 'N/A')}")
                    
                    return False
                else:
                    print(f"  Field {field} is equal")
                    
            except Exception as e:
                print(f"[DEBUG] Error comparing field {field}: {e}")
                return False
        
        print("\n[DEBUG] All fields match, configs are considered equal")
        return True
    
    def __hash__(self):
        # Create a hash based on all fields except hf_config
        hash_values = []
        for field in self.__dataclass_fields__:
            if field == 'hf_config':
                # Only include model_type from hf_config in the hash
                hf_type = getattr(self.hf_config, 'model_type', None)
                hash_values.append(hash(hf_type) if hf_type is not None else 0)
            else:
                value = getattr(self, field)
                try:
                    hash_values.append(hash(value))
                except TypeError:
                    # Handle unhashable types by converting to string
                    hash_values.append(hash(str(value)))
        
        # Combine all hash values
        return hash(tuple(hash_values))
