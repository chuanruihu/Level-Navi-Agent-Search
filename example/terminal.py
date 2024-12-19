import os
import sys
import json

from termcolor import colored
from dotenv import load_dotenv

current_dir = os.path.dirname(__file__)  
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.serve import VllmServer
from src.plugins import QihooWebSearch, BingSearch
from src.actions import ActionExecutor, SearchAction, SelectAction
from src.ai_search import ResultSaves, PlanningAgent, SearcherAgent, SearchDistributor

env_path = os.path.join(project_root, 'config', '.env')
load_dotenv(dotenv_path=env_path)

def main():
    # llm = VllmServer(
    #     model_name="qwen2.5-7b-instruct",
    #     api_key="",
    #     api_base="",
    # )
    llm = VllmServer(
        model_name=os.getenv("MODEL_NAME"),
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
    )
    tool_info, tool_map = ActionExecutor.get_tool_info(SearchAction, SelectAction)
    agent = PlanningAgent(
        llm,
        SearchDistributor(
            searcher_type=SearcherAgent,
            llm=llm,
            max_turn=3,
            topk=6,
            # searcher_class=BingSearch,
            searcher_class=QihooWebSearch,
            tool_info=tool_info,
            tool_map=tool_map,
        ),
        max_turn=4,
        debug=True
    )

    query = "今年TGA年度最佳游戏提名作品的具体发售时间分别是什么？"
    agent_result = None
    for agent_result in agent.stream_chat(query):
        pass
    print("=="*20,"最终答案","=="*20)
    print(colored(agent_result.response,"light_green"))

if __name__ == "__main__":
    main()
