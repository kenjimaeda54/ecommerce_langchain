import os
from langsmith import traceable

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, content, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv


MAX_INTERACTIONS = 10

load_dotenv()

apiKeyOpenRouter = os.getenv('OPEN_ROUTER_API_KEY')


@tool
def get_price_by_product_name(product_name: str) -> float:
    """
    Get the price of a product by its name.
    Valid product names are: laptop, phone, tablet.
    """
    prices = {
        "laptop": 1000,
        "phone": 500,
        "tablet": 300
    }
    return prices.get(product_name.lower(),0)

@tool
def  apply_discount_by_tier(price: float,tier: str) -> float:
    """
    Apply discount by tier.
    """
    discount_tier = { "silver": 15, "gold": 25, "bronze": 30 }
    discount = discount_tier.get(tier.lower(),0)

    return round(price * (1 - discount / 100 ), 2)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [apply_discount_by_tier, get_price_by_product_name]
    tool_dict = {t.name: t for t in tools}
    llm = init_chat_model(
        model="openai/gpt-4o-mini",
        model_provider="openai",
        api_key=apiKeyOpenRouter,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content="You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
        ),
        HumanMessage(content=question)

    ]


    for interaction in range(1,MAX_INTERACTIONS + 1):
        ai_message = llm_with_tools.invoke(messages)

        tools_calls = ai_message.tool_calls

        if not tools_calls:
            return ai_message.content

        tool_call = tools_calls[0]
        tool_name =  tool_call.get("name")
        tool_call_id = tool_call.get("id")
        tool_args = tool_call.get("args", {})

        tool_to_use = tool_dict.get(tool_name)

        if tool_to_use is None:
            raise  ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)


        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id,
            )
        )


    return None

if __name__ == "__main__":
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print(result)