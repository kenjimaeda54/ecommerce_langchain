import os

from langsmith import traceable

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from dotenv import load_dotenv


# Mesmos limites conceitos do hello.py (10 voltas Thought -> Action -> Observation)
MAX_ITERATIONS = 10

load_dotenv()

apiKeyOpenRouter = os.getenv("OPEN_ROUTER_API_KEY")


# --- Ferramentas (idênticas ao hello.py) ---
# O create_agent descobre e chama estas funções sozinho; não precisamos
# escrever o loop que executa tool_calls nem anexar ToolMessage manualmente.
@tool
def get_price_by_product_name(product_name: str) -> float:
    """
    Get the price of a product by its name.
    Valid product names are: laptop, phone, tablet.
    """
    prices = {
        "laptop": 1000,
        "phone": 500,
        "tablet": 300,
    }
    return prices.get(product_name.lower(), 0)


@tool
def apply_discount_by_tier(price: float, tier: str) -> float:
    """
    Apply discount by tier.
    """
    discount_tier = {"silver": 15, "gold": 25, "bronze": 30}
    discount = discount_tier.get(tier.lower(), 0)

    return round(price * (1 - discount / 100), 2)


@traceable(name="LangChain create_agent")
def run_agent(question: str):
    tools = [apply_discount_by_tier, get_price_by_product_name]

    llm = init_chat_model(
        model="openai/gpt-4o-mini",
        model_provider="openai",
        api_key=apiKeyOpenRouter,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )

    system_prompt = (
        "You are a helpful shopping assistant. "
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
    )

    # create_agent monta o grafo do agente (LangGraph) que já faz o loop ReAct
    # pronto: chama o modelo, executa as ferramentas, devolve o resultado como
    # ToolMessage e repete até não haver mais tool_calls. Substitui, de uma vez,
    # o bind_tools + o for-loop manual que estavam em hello.py.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    # recursion_limit trava o número de voltas Thought -> Action -> Observation
    # (equivalente ao MAX_INTERACTIONS do loop manual em hello.py).
    config: RunnableConfig = {"recursion_limit": MAX_ITERATIONS}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )
    # O estado retornado traz a lista de mensagens; a resposta final é a última.
    return result["messages"][-1].content


if __name__ == "__main__":
    result = run_agent(
        "What is the price of a laptop after applying a gold discount?"
    )
    print(result)
