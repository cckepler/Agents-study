# 假定 llm_client.py 中的 HelloAgentsLLM 类已经定义好
# from llm_client import HelloAgentsLLM
import ast
import re

from ..prompt.prompt import PLANNER_PROMPT_TEMPLATE, EXECUTOR_PROMPT_TEMPLATE


class PlanAndSolveAgent:
    def __init__(self, llm_client, tool_executor):
        """
        初始化智能体，同时创建规划器和执行器实例。
        """
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client, tool_executor)

    def run(self, question: str):
        """
        运行智能体的完整流程:先规划，后执行。
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)

        # 检查计划是否成功生成
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return

        # 2. 调用执行器执行计划
        final_answer = self.executor.execute(question, plan)

        #print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
        return final_answer


class Planner:
    def __init__(self,llm_client):
        self.llm_client = llm_client
    def plan(self,question:str) -> list[str]:
        """
        根据用户问题生成一个行动计划。
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)

        messages = [{"role": "user", "content": prompt}]

        response = self.llm_client.think(messages=messages) or ""

        print(f"计划已生成:\n{response}")

        # 解析LLM输出的列表字符串
        try:
            # 找到```python和```之间的内容
            plan_str = response.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"解析计划时出错: {e}")
            print(f"原始响应: {response}")
            return []
        except Exception as e:
            print(f"解析计划时发生未知错误: {e}")
            return []


class Executor:
    def __init__(self, llm_client, tool_executor):
        self.llm_client = llm_client
        self.tool_executor = tool_executor

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action"""
        thought_match = re.search(r"Thought:\s*(.*)", text)
        action_match = re.search(r"Action:\s*(.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称与输入"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划逐步执行，每一步可以调用工具（ReAct机制融合版）。
        """
        history = ""
        print("\n--- 正在执行计划 ---")
        tools_desc = self.tool_executor.getAvailableTools()

        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i + 1}/{len(plan)}: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                tool_executor=tools_desc,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages) or ""
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"思考: {thought}")

            if not action:
                print(f"未检测到有效 Action，跳过该步骤。")
                history += f"步骤 {i + 1}: {step}\n结果: 无效响应\n\n"
                continue

            print(f"行动: {action}")

            # 处理工具调用逻辑
            if action.startswith("Finish"):
                # 支持跨行匹配
                match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)  # 支持跨行匹配
                if match:
                    step_result = match.group(1).strip()
                    #step_result = match.group(1).strip() if match else "未知结果"
                    print(f"步骤 {i + 1} 完成，无需调用工具。结果: {step_result}")

            else:
                # 工具调用逻辑
                tool_name, tool_input = self._parse_action(action)
                if not tool_name or not tool_input:
                    print(f"工具调用格式错误，无法解析。")
                    step_result = "无效工具调用"
                else:
                    tool_func = self.tool_executor.getTool(tool_name)
                    if not tool_func:
                        print(f"工具 {tool_name} 未注册")
                        step_result = f"工具 {tool_name} 不存在"
                    else:
                        # 真正执行工具
                        observation = tool_func(tool_input)
                        print(f"工具 {tool_name} 执行结果: {observation}")
                        step_result = observation

            # 记录历史
            history += f"步骤 {i + 1}: {step}\n思考: {thought}\n行动: {action}\n结果: {step_result}\n\n"

        final_answer = step_result
        return final_answer