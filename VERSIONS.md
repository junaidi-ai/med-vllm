# Nano vLLM Medical Configuration Version History

## Version 0.1.0 (Initial Release)
**Release Date**: TBA

### Configuration Format
```json
{
  "config_version": "0.1.0",
  "model_type": "bert",
  "model": "path/to/model",
  "max_medical_seq_length": 512,
  "medical_specialties": ["cardiology", "neurology"],
  "anatomical_regions": ["head", "chest"],
  "enable_uncertainty_estimation": true,
  "max_num_batched_tokens": 16384,
  "max_num_seqs": 512,
  "max_model_len": 4096,
  "gpu_memory_utilization": 0.9,
  "tensor_parallel_size": 1,
  "enforce_eager": false,
  "kvcache_block_size": 256,
  "num_kvcache_blocks": -1
}
```

### Migration Paths

#### From Pre-0.1.0 to 0.1.0
- Added `config_version` field (required)
- Standardized medical parameters with default values
- Added validation for all configuration parameters

### Deprecation Notices
- None in initial release

### Upgrade Instructions
For new configurations, use the latest format shown above. Existing configurations without a version will be automatically migrated to version 0.1.0 with default values for any missing fields.

## Future Versions
This section will document any future changes to the configuration format, including:
- New parameters
- Deprecated parameters
- Breaking changes
- Migration paths from previous versions
