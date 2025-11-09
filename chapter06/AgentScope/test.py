import asyncio
import base64
import os
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory, LongTermMemoryBase
from agentscope.message import (
    Msg,
    Base64Source,
    TextBlock,
    ImageBlock,
)
from dotenv import load_dotenv
load_dotenv()
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit
LongTermMemoryBase
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"current_dir:{current_dir}")
# 回到上两级目录
project_root = os.path.dirname(os.path.dirname(current_dir))
print(f"project_root:{project_root}")
image_path = os.path.join(project_root, "images", "1.png")
print(f"image_path:{image_path}")
with open(image_path, "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode("utf-8")


msg = Msg(
    name="xjx",
    role="user",
    content=[
        TextBlock(
            type="text",
            text="这张图片是什么",
        ),
        # ImageBlock(
        #     type="image",
        #     source=URLSource(
        #         type="url",
        #         url="https://picture-1351272937.cos.ap-guangzhou.myqcloud.com/public/1976236295399251969/2025-10-09_4RdC4S5PKpGvgbYR.webp"
        #     )
        # )
        ImageBlock(
            type="image",
            source=Base64Source(
                type="base64",
                media_type="image/jpeg",
                data=img_base64,
            ),
        ),
    ],
)

friday = ReActAgent(
    name="Friday",
    sys_prompt="你是一个名为 Friday 的有用助手",
    model=DashScopeChatModel(
        model_name="qwen3-vl-flash",
        api_key=os.getenv("LLM_API_KEY"),
    ),
    formatter=DashScopeChatFormatter(),  # 用于 user-assistant 对话的格式化器
    memory=InMemoryMemory(),
    toolkit=Toolkit(),
)


async def test_friday(message) -> None:
    print(message)
    """测试 Friday 智能体是否能够响应消息。"""
    res = await friday(message)
    assert res is not None
    assert res.get_text_content() is not None
    print(res)


asyncio.run(test_friday(msg))
