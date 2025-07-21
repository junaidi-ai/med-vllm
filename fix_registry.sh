#!/bin/bash

# Create a backup of the original file
cp medvllm/engine/model_runner/registry.py medvllm/engine/model_runner/registry.py.bak

# 1. Add force parameter to register method
sed -i 's/def register(\s*self,\s*name:\s*str,/def register(self, name: str, force: bool = False,/' medvllm/engine/model_runner/registry.py

# 2. Update the duplicate check to use the force parameter
sed -i '/if name in self._models:/ {
    N
    s/if name in self._models:\s*\n\s*raise ValueError/if name in self._models and not force:\n            return\n        if name in self._models:\n            del self._models[name]\n        if name in self._models:\n            raise ValueError/
}' medvllm/engine/model_runner/registry.py

# 3. Update _register_default_models to accept force parameter
sed -i 's/def _register_default_models(self) -> None:/def _register_default_models(self, force: bool = False) -> None:/' medvllm/engine/model_runner/registry.py

# 4. Update register calls in _register_default_models to pass force parameter
sed -i '/def _register_default_models/,/def get_registry/ s/self\\.register(/self.register(force=force, /g' medvllm/engine/model_runner/registry.py

# 5. Update get_registry to pass force=False
sed -i 's/registry\\._register_default_models()/registry._register_default_models(force=False)/' medvllm/engine/model_runner/registry.py

echo "✅ Registry file has been updated successfully. Original saved as registry.py.bak"
