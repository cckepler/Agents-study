import os
from openai import  OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
# 加载环境变量
load_dotenv()

class HelloAgentsLLM:
    def __init__(self,model:str=None,apiKey:str=None,baseUrl:str=None,timeout:int=None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.apiKey = apiKey or os.getenv("LLM_API_KEY")
        self.baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        if not all([self.model, self.apiKey, self.baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")
        self.client = OpenAI(
            api_key=self.apiKey,
            base_url=self.baseUrl,
            timeout=self.timeout,
        )

    def think(self,messages:List[Dict[str,str]],temperature:float=0.7) -> Optional[str]:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            print("LLM Response Success")
            #处理流式响应
            collected_content = []
            """
            chunk
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": "你好"
                        },
                        "index": 0,
                        "finish_reason": null
                    }
                ]
            }
            """
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                collected_content.append(content)
                #print(content, end="", flush=True)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return None


# if __name__ == "__main__":
    # try:
    #     llmClient  = HelloAgentsLLM()
    #     exampleMessages = [
    #         {"role": "system", "content": "You are a helpful assistant that writes Python code."},
    #         {"role": "user", "content": "写一个快速排序算法"}
    #     ]
    #     response = llmClient.think(exampleMessages)
    #     print(response)
    # except ValueError as e:
    #     print(f"初始化LLM时发生错误: {e}")
    # toolExecutor = ToolExecutor()
    # search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    # toolExecutor.registerTool("Search",search_description,search)
    # print("\n---可用工具---")
    # print(toolExecutor.getAvailableTools())
    #
    # print("\n----获取指定工具---")
    # print(toolExecutor.getTool("Search"))
    #
    # print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    # tool_name = "Search"
    # query = "英伟达最新的GPU型号是什么"
    # tool_function = toolExecutor.getTool(tool_name)
    # if tool_function:
    #     observation = tool_function(query)
    #     print(observation)
    # else:
    #     print(f"工具 {tool_name} 不存在")




