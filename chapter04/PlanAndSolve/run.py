from chapter04.PlanAndSolve.planAndSlove import PlanAndSolveAgent
from chapter04.llm_client import HelloAgentsLLM
from chapter04.tools import ToolExecutor, search

if __name__ == "__main__":

    try:
        llm_client = HelloAgentsLLM()
        tool_executor = ToolExecutor()
        search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
        tool_executor.registerTool("Search", search_desc, search)
        agent = PlanAndSolveAgent(llm_client,tool_executor)

        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        agent.run(question)
    except ValueError as e:
        print(e)