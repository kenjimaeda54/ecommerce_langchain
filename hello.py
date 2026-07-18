from idlelib.debugger_r import tracebacktable

from langchain_core.tools import tool


@tool
def get_price_by_product(product: str) -> float:
    """"
     Look up price of a product in the catalog
    """
    prices = {"lapptop": 1000, "phone": 500, "tablet": 300}
    return prices.get(product, 0)

@tool
def apply_discount(tier_discount: str,price: float) -> float:
    """
     Look up discount price for a tier discount
    """
    discount_percentage = {"gold": 25, "silver": 10, "bronze": 15}
    discount = discount_percentage.get(tier_discount, 0)
    return round(price * (1 - discount / 100), 2)


@tracebacktable
def run_agent(question: str):
    pass

if __name__ == "__main__":
     run_agent("what is the price of a lapptop with gold discount?")