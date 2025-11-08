from PlanAndSolve.planAndSlove import Planner, PlanAndSolveAgent
from ReActAgent.llm_client import HelloAgentsLLM
from ReActAgent.react_agent import ReActAgent
from tools.ToolExecutor import ToolExecutor
from tools.google_search import search

# 测试
if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    # ReAct模式
    # agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    # question = "苹果最新的手机是哪一款？它的主要卖点是什么？"
    # agent.run(question)
    # PlanAndSolve模式
    pasAgent = PlanAndSolveAgent(llm_client=llm,tool_executor=tool_executor)
    question = "问题: 一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    response = pasAgent.run(question)
    print(f"\n--- 任务完成 ---\n最终答案: {response}")
    # 比较使用工具和普通LLM的输出
    # exampleMessages = [
    #         {"role": "system", "content": "You are a helpful assistant that writes Python code."},
    #         {"role": "user", "content": question}
    #     ]
    # response = llm.think(exampleMessages)
    # print(response)
