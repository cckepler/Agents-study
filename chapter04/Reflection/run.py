from chapter04.llm_client import HelloAgentsLLM
from chapter04.Reflection.Reflection import ReflectionAgent


if __name__ == '__main__':
    try:
        llm_client = HelloAgentsLLM()
    except Exception as e:
        print(f"初始化LLM客户端时出错: {e}")
        exit()

    # 2. 初始化 Reflection 智能体，设置最多迭代2轮
    agent = ReflectionAgent(llm_client, max_iterations=3)

    # 3. 定义任务并运行智能体
    task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    res = agent.run(task)
    print(res)