class Registry:
    mapping = {
        "mas_name_mapping": {},
        "memory_name_mapping": {},
        "builder_name_mapping": {},
        "env_name_mapping": {},
        "runner_name_mapping": {},
        "state": {},
        "paths": {}
    }
    
    @classmethod
    def register(cls, name, obj):
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def register_mas(cls, name):

        def wrap(mas_cls):
            from memmaster.mas_core.base_memory_mas import BaseMemoryMAS
            assert issubclass(
                mas_cls, BaseMemoryMAS
            ), "All MAS must inherit BaseMemoryMAS class"
            if name in cls.mapping["mas_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["mas_name_mapping"][name]
                    )
                )
            cls.mapping["mas_name_mapping"][name] = mas_cls
            return mas_cls

        return wrap

    @classmethod
    def register_memory(cls, name):

        def wrap(memory_cls):
            from memmaster.mas_core.base_centralized_memory import BaseCentralizedMemory
            assert issubclass(
                memory_cls, BaseCentralizedMemory
            ), "All memory must inherit BaseCentralizedMemory class"
            if name in cls.mapping["memory_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["memory_name_mapping"][name]
                    )
                )
            cls.mapping["memory_name_mapping"][name] = memory_cls
            return memory_cls

        return wrap
    
    @classmethod
    def register_path(cls, name, path):

        assert isinstance(path, str), "All path must be str."
        if name in cls.mapping["paths"]:
            raise KeyError("Name '{}' already registered.".format(name))
        cls.mapping["paths"][name] = path
    
    @classmethod
    def register_builder(cls, name):

        def wrap(builder_cls):
            from data.base_builder import BaseDataBuilder

            assert issubclass(
                builder_cls, BaseDataBuilder
            ), "All builders must inherit BaseDatasetBuilder class, found {}".format(
                builder_cls
            )
            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["builder_name_mapping"][name]
                    )
                )
            cls.mapping["builder_name_mapping"][name] = builder_cls
            return builder_cls

        return wrap
    
    @classmethod
    def register_env(cls, name):

        def wrap(env_cls):
            from data.base_env import BaseEnv

            assert issubclass(
                env_cls, BaseEnv
            ), "All environments must inherit BaseEnv class, found {}".format(
                env_cls
            )
            if name in cls.mapping["env_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["env_name_mapping"][name]
                    )
                )
            cls.mapping["env_name_mapping"][name] = env_cls
            return env_cls

        return wrap

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)
    
    @classmethod
    def get_env_class(cls, name):
        return cls.mapping["env_name_mapping"].get(name, None)
    
    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)

    @classmethod
    def get_mas_class(cls, name):
        return cls.mapping["mas_name_mapping"].get(name, None)
    
    @classmethod
    def get_memory_class(cls, name):
        return cls.mapping["memory_name_mapping"].get(name, None)   
    
    @classmethod
    def get_runner_class(cls, name):
        return cls.mapping["runner_name_mapping"].get(name, None)
    
registry = Registry()