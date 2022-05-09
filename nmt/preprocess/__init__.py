import importlib
import inspect
import pkgutil
import os

from collections import OrderedDict
from nmt.common.utils import parse_yaml
from nmt.preprocess import task
from nmt.preprocess.task import Task, log
from nmt.tokenize import Tokenizer


class PreProcessor(object):
    def __init__(self, config_path: str):
        self.config = parse_yaml(config_path)
        self.config_task_names = OrderedDict({
            dic["name"]: dic.get("instance", {}) for dic in self.config["task_list"]
        })
        self.load_tasks()
        self.validate_config_task()

    def load_tasks(self):
        task_cls_dict = {}

        for importer, modname, ispkg in pkgutil.iter_modules(task.__path__):
            if not ispkg:
                task_module = importlib.import_module("." + modname, package='nmt.preprocess.task')
                cls_members = [
                    (name, cls) for name, cls in inspect.getmembers(task_module, inspect.isclass) 
                    if cls.__module__ not in [Task.__module__, Tokenizer.__module__]
                ]

                for name, cls in cls_members:
                    if name in self.config_task_names:
                        task_cls_dict[name] = getattr(task_module, name)

        self.task_cls_dict = task_cls_dict

    def validate_config_task(self):
        for task_name in self.config_task_names:
            msg = f"'{task_name}' is not available task\nValids: {self.task_cls_dict.values()}"
            assert task_name in self.task_cls_dict, msg

    def create_pipelines(self):
        task_pipelines = []

        base_task_config = {}
        base_task_config["src_lang"] = self.config["src_lang"]
        base_task_config["tgt_lang"] = self.config["tgt_lang"]
        base_task_config["src_corpus_path"] = self.config["src_corpus_path"]
        base_task_config["tgt_corpus_path"] = self.config["tgt_corpus_path"]

        for task_name, task_instance in self.config_task_names.items():
            assert task_name in self.task_cls_dict
            task_cls = self.task_cls_dict[task_name]
            task_config = dict(base_task_config)

            for key in task_instance:
                task_config[key] = task_instance[key]

            task_pipelines.append((task_name, task_cls(config=task_config)))

        return task_pipelines

    def execute(self):
        task_pipelines = self.create_pipelines()
        src_input_path = self.config["src_corpus_path"]
        tgt_input_path = self.config["tgt_corpus_path"]

        assert os.path.dirname(src_input_path) == os.path.dirname(tgt_input_path)
        base_dir = os.path.dirname(src_input_path)

        src_base_name = os.path.basename(src_input_path)
        tgt_base_name = os.path.basename(tgt_input_path)

        prev_suffix = None
        prev_task_idx = None

        for idx, (task_name, task_instance) in enumerate(task_pipelines, 1):
            task_idx = str(idx).zfill(2)
            if not prev_suffix:
                prev_src_name = src_base_name
                prev_tgt_name = tgt_base_name
            else:
                prev_src_name = f"{src_base_name}.Task{prev_task_idx}.{prev_suffix}"
                prev_tgt_name = f"{tgt_base_name}.Task{prev_task_idx}.{prev_suffix}"

            prev_src_path = os.path.join(base_dir, prev_src_name)
            prev_tgt_path = os.path.join(base_dir, prev_tgt_name)

            next_suffix = task_name
            next_task_idx = task_idx
            next_src_name = f"{src_base_name}.Task{next_task_idx}.{next_suffix}"
            next_tgt_name = f"{tgt_base_name}.Task{next_task_idx}.{next_suffix}"

            next_src_path = os.path.join(base_dir, next_src_name)
            next_tgt_path = os.path.join(base_dir, next_tgt_name)

            if not task_instance.is_done(next_src_path, next_tgt_path):
                task_instance.run(src_input_path=prev_src_path,
                                  tgt_input_path=prev_tgt_path,
                                  src_output_path=next_src_path,
                                  tgt_output_path=next_tgt_path)

                err_msg = f"'{next_src_path}' does not exists\n'{next_tgt_path}' does not exists"
                assert task_instance.is_done(next_src_path, next_tgt_path), err_msg
            else:
                log.info(f"Skipping '{task_name}', because it is already done")

            prev_suffix = task_name
            prev_task_idx = task_idx