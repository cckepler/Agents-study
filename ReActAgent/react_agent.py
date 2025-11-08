# react_agent.py
import re
import sys
from pathlib import Path

react_agent_dir = Path(__file__).parent
if str(react_agent_dir) not in sys.path:
    sys.path.insert(0, str(react_agent_dir))

project_root = react_agent_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm_client import HelloAgentsLLM
from prompt.prompt import REACT_PROMPT_TEMPLATE
from tools.ToolExecutor import ToolExecutor
from tools.google_search import search
class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。"""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None
    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = []  # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            # 将历史记录加入prompt todo 持久化历史记录
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # 3. 解析LLM输出
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"思考",thought)

            if not action:
                print(f"未识别到有效行动")
                break

            print(f"行动:{action}")

            if action.startswith("Finish"):
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案：{final_answer}")
                return final_answer

            tool_name,tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # todo 处理无效工具->llm 幻觉调用
                print(f"行动格式错误，无法解析工具名称或输入")
                continue

            # 4.执行工具
            tool_func = self.tool_executor.getTool(tool_name)
            if not tool_func:
                print(f"错误:工具 {tool_name} 未注册")
                continue
            else:
                observation = tool_func(tool_input)
                print(f"工具 {tool_name} 执行结果: {observation}")

            # 5.增加当前历史记录
            self.history.append(f"思考: {thought}\n行动: {action}\n结果: {observation}")

        print("最大步数已达，流程终止...")
        return None


