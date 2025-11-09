# implement the multi-agent system
# Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. EMNLP 2024
import asyncio
import os

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter, DashScopeMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.pipeline import MsgHub
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
question = "两个圆外切且没有相对滑动。圆A的半径是圆B半径的1/3。圆A绕圆B滚动一圈回到起点。圆A总共会旋转多少次？"


# 创建单个智能体 ReAct模式
def create_agent(name:str)->ReActAgent:
    return ReActAgent(
        name=name,
        sys_prompt=f"你是一个名为 {name} 的辩论者。你好，欢迎来到"
                   "数学辩论比赛。我们的目标是找到正确答案，因此你没有必要完全同意对方"
                   f"的观点。辩论问题如下所述：{question}",
        model=DashScopeChatModel(
            model_name="qwen-flash",
            api_key=os.getenv("LLM_API_KEY"),
        ),
        formatter=DashScopeChatFormatter(),  # 用于 user-assistant 对话的格式化器
        memory=InMemoryMemory(), # 内存存储 -> 可以自定义
    )

Alex,Jack = [create_agent(name) for name in ["Alex","Jack"]]

referee = ReActAgent(
    name="Aggregator",
    sys_prompt=f"""你是一个主持人。将有两个辩论者参与辩论比赛。他们将就以下话题提出观点并进行讨论：
``````
{question}
``````
在每轮讨论结束时，你将评估辩论是否结束，以及问题正确的答案。""",
    model=DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.getenv("LLM_API_KEY"),
        stream=False,
    ),
    # 使用多智能体格式化器，因为主持人将接收来自多个agent的消息
    formatter=DashScopeMultiAgentFormatter(),
)

# 结构化输出模型
class JudgeModel(BaseModel):
    finished: bool = Field(default=False, description="是否辩论结束")
    correct_answer: str | None = Field(
        default=None,
        description="辩论话题的正确答案，仅当辩论结束时提供该字段。否则保留为 None。",
    )

async def judge_debate() -> None:
    while True:
        # 使用MsgHub传递消息
        async with MsgHub(participants=[Alex,Jack,referee]) as hub:
            await Alex(
                Msg(
                    name = "user",
                    content = "你是正方，请表达你的观点。",
                    role = "user",
                ),
            )
            await Jack(
                Msg(
                    name = "user",
                    content = "你是反方。你不同意正方的观点。请表达你的观点和理由。",
                    role = "user",
                ),
            )

        msg_judge = await referee(
            Msg(
                name = "user",
                content = "现在你已经听到了他们的辩论，现在判断辩论是否结束，以及你能得到正确答案吗？",
                role = "user",
            ),
            structured_model=JudgeModel,
        )

        if msg_judge.metadata.get("finished",False):
            print(
                "\n辩论结束，正确答案是：",
                msg_judge.metadata.get("correct_answer"),
            )
            break

asyncio.run(judge_debate())
