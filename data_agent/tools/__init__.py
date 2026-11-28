"""工具注册表：加工具只需要动这一个文件的两行。"""
import traceback

from ..sandbox import RUN_PYTHON_SCHEMA, run_python
from .tabular import LOAD_DATA_SCHEMA, load_data, list_files, LIST_FILES_SCHEMA

REGISTRY = {
    "load_data": load_data,
    "run_python": run_python,
    "list_files": list_files
}

SCHEMAS = [LOAD_DATA_SCHEMA, RUN_PYTHON_SCHEMA,LIST_FILES_SCHEMA]


def dispatch(name: str, args: dict) -> str:
    try:
        return REGISTRY[name](**args)
    except KeyError:
        return f"未知工具: {name}"
    except TypeError as exc:
        return f"工具参数不匹配: {exc}"
    except Exception:
        return "工具执行异常:\n" + traceback.format_exc(limit=3)