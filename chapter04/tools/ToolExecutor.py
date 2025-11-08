from typing import Dict,Any

# 工具管理
class ToolExecutor:
    # 存储所有工具
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        注册一个工具。
        :param name: 工具的名称，用于在执行时引用。
        :param description: 工具的简介
        :param func: 工具的函数实现，用于执行具体的操作。
        """
        if name in self.tools:
            print(f"工具 {name} 已存在,将被覆盖")

        self.tools[name] = {
            "description": description,
            "func": func
        }
        print(f"工具 {name} 已注册")

    # 指定获取单个工具
    def getTool(self, name: str) -> callable:
        """
        获取指定名称的工具。
        :param name: 要获取的工具名称。
        :return: 执行方法。
        """
        return self.tools.get(name, {}).get("func")

    # 获取全部工具信息
    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
