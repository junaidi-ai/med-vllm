def patch_registry():
    """Patch the registry to handle duplicate registrations gracefully."""
    import sys

    from medvllm.engine.model_runner.registry import ModelRegistry

    original_register = ModelRegistry.register

    def patched_register(
        self,
        name: str,
        model_type=None,
        model_class=None,
        config_class=None,
        description: str = "",
        tags=None,
        loader=None,
        force: bool = False,
        **parameters,
    ):
        """Patched register method that skips existing registrations by default."""
        if tags is None:
            tags = []

        with self._lock:
            if name in self._models and not force:
                print(
                    f"Model '{name}' is already registered. Skipping...",
                    file=sys.stderr,
                )
                return
            return original_register(
                self,
                name=name,
                model_type=model_type,
                model_class=model_class,
                config_class=config_class,
                description=description,
                tags=tags,
                loader=loader,
                **parameters,
            )

    # Apply the patch
    ModelRegistry.register = patched_register
    print("Registry patched to handle duplicate registrations gracefully", file=sys.stderr)


# Apply the patch when this module is imported
patch_registry()
