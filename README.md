# E-commerce LangChain Agent

Um **agente de IA em Python** construído com [LangChain](https://www.langchain.com/) que atua como um assistente de compras. Ele usa o padrão de arquitetura **ReAct (Reasoning + Acting)** para consultar um catálogo de produtos e aplicar descontos por meio de ferramentas (tools), em vez de "inventar" respostas.


---

## 🧠 Sobre a arquitetura: ReAct (Reasoning + Acting)

ReAct (Yao et al., 2022) é um padrão onde o modelo de linguagem **alterna** raciocínio e ação:

```
Thought (penso)  →  Action (uso uma ferramenta)  →  Observation (vejo o resultado)  →  repetir até a resposta
```

Em vez de responder direto (o que levaria a alucinações), o agente **chama ferramentas reais** para obter dados e só então conclui.

### Como isto se traduz no código (`hello.py`)

| Etapa ReAct | Implementação |
|-------------|---------------|
| **Thought** (decisão do modelo) | `llm_with_tools.invoke(messages)` |
| **Action** (chamada de ferramenta) | `ai_messages.tool_calls` → `get_price_by_product_name` / `apply_discount_by_tier` |
| **Observation** (resultado) | `ToolMessage(content=str(observation), tool_call_id=...)` |
| **Scratchpad / memória** | a lista `messages`, que acumula `AIMessage` + `ToolMessage` a cada volta |
| **Limite de ciclos** | `MAX_INTERACTIONS = 10` (máx. 10 voltas Thought→Action→Observation) |

**Exemplo real** para *"preço de um laptop com desconto gold"*:
1. **Thought:** preciso do preço → **Action:** `get_price_by_product_name("laptop")`
2. **Observation:** `1000` é adicionado ao histórico
3. **Thought:** aplico gold sobre 1000 → **Action:** `apply_discount_by_tier(1000, "gold")`
4. **Observation:** `750.0`
5. **Thought:** tenho tudo → sem `tool_calls` → retorna a resposta final

---

## 🛠️ Stack / Tecnologias

- **Python 3.10+** — linguagem do projeto (back-end / script de terminal)
- **LangChain** (`langchain`, `langchain-core`) — orquestração do agente e ferramentas
- **langchain-openai** — cliente de modelos compatíveis com a API OpenAI
- **OpenRouter** — roteador de LLMs usado como endpoint OpenAI-compatible (`base_url`)
- **Modelo:** `openai/gpt-4o-mini` (via OpenRouter)
- **LangSmith** — observabilidade/tracing das execuções do agente
- **python-dotenv** — carregamento de variáveis de ambiente do `.env`
- **black** + **isort** — formatação e ordenação de imports

> 💡 **Por que LangChain e não React?** O objetivo é fazer um LLM "pensar" e usar ferramentas (preço/desconto). Isso é lógica de IA/back-end. React só faria sentido se você quisesse uma **interface de chat visual** na frente deste agente (aí sim: React + este Python como API).

---

## 📁 Estrutura do projeto

```
ecommerce_langchain/
├── hello.py          # Código principal: definição das tools + loop ReAct
├── pyproject.toml    # Dependências e metadados (gerenciado via uv)
├── uv.lock           # Lockfile de dependências
├── .env              # Variáveis de ambiente (API keys) — NÃO versionar
├── .gitignore        # Ignora .venv, .env, __pycache__
└── README.md         # Este arquivo
```

---

## ⚙️ Pré-requisitos

- Python **3.10 ou superior**
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes recomendado) — ou `pip` tradicional
- Chaves de API:
  - `OPEN_ROUTER_API_KEY` (OpenRouter)
  - `LANGSMITH_API_KEY` (LangSmith, opcional p/ tracing)

---

## 🚀 Como iniciar o projeto

### 1. Clonar e entrar na pasta
```bash
git clone <seu-repo> ecommerce_langchain
cd ecommerce_langchain
```

### 2. Criar o ambiente virtual e instalar dependências

O projeto já possui um **`uv.lock`** (lockfile gerado pelo uv). O fluxo recomendado aproveita esse arquivo para garantir versões **reproduzíveis**.

Com **uv** (recomendado — usa o `uv.lock`):
```bash
# Cria o .venv e instala exatamente as versões do uv.lock
uv sync

# Ativar o ambiente (opcional, o 'uv run' já faz isso sozinho)
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

> 💡 `uv sync` lê o `uv.lock` e reproduz o mesmo ambiente em qualquer máquina. Se você alterar o `pyproject.toml`, rode `uv lock` para atualizar o lockfile e depois `uv sync`.

Rodando o projeto com uv (sem precisar ativar o venv manualmente):
```bash
uv run python hello.py
```

Outros comandos úteis do uv:
```bash
uv lock            # recalcula o uv.lock após mudar o pyproject.toml
uv add black       # adiciona uma dependência e atualiza lock+venv
uv remove black    # remove uma dependência
uv pip list        # lista o que está instalado no .venv
```

Ou com **pip** tradicional (ignora o lockfile):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Configurar as variáveis de ambiente

Crie um arquivo `.env` na raiz (ele já existe neste repo, mas **nunca o comite**):
```env
OPEN_ROUTER_API_KEY=sua-chave-aqui
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=sua-chave-langsmith
LANGSMITH_PROJECT=search_job
```

### 4. Rodar o agente

```bash
python hello.py
```

Saída esperada (exemplo):
```
The price of a laptop after a gold discount is $750.00.
```

---

## 🔧 Como funciona cada etapa (detalhado)

### 1. Carregamento de ambiente
```python
load_dotenv()
apiKeyOpenRouter = os.getenv('OPEN_ROUTER_API_KEY')
```
Lê as chaves do `.env` para não expor segredos no código.

### 2. Definição das ferramentas (`@tool`)
Duas funções Python viram "ferramentas" que o LLM pode chamar:
- `get_price_by_product_name(product_name)` → retorna o preço real do catálogo (`laptop`, `phone`, `tablet`).
- `apply_discount_by_tier(price, tier)` → aplica desconto (`silver` 15%, `gold` 25%, `bronze` 30%).

A docstring de cada função é **o que o modelo lê** para decidir quando usá-la — por isso ser preciso na docstring importa.

### 3. Inicialização do modelo
```python
llm = init_chat_model(
    model="openai/gpt-4o-mini",
    model_provider="openai",
    api_key=apiKeyOpenRouter,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)
```
`bind_tools` "ensina" ao modelo quais ferramentas existem. `temperature=0` deixa as respostas determinísticas.

### 4. O prompt do sistema (disciplina de ReAct)
O `SystemMessage` força o raciocínio em etapas:
1. Nunca adivinhe preço → chame a ferramenta primeiro
2. Só aplique desconto **depois** de ter o preço real
3. Nunca calcule na mão → use a ferramenta
4. Se faltar o tier → pergunte, não assuma

### 5. O loop ReAct (`for interaction in range(...)`)
A cada volta:
1. O modelo recebe todo o histórico (`messages`) e decide o próximo passo.
2. Se houver `tool_calls`, executa a ferramenta e anexa o resultado como `ToolMessage`.
3. Se **não** houver `tool_calls`, o modelo já tem a resposta → retorna o conteúdo.
4. O modelo **não tem memória própria**: o histórico é reenviado toda volta (isso é a "caderneta" do ReAct).

---

## 📊 Observabilidade com LangSmith

Com `LANGSMITH_TRACING=true`, cada execução é registrada no [LangSmith](https://smith.langchain.com/). O decorator:
```python
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
```
marca o início do trace, permitindo inspecionar cada Thought/Action/Observation no dashboard.

---

## 🧪 Modificando o agente

Para testar outras perguntas, altere a chamada no final de `hello.py`:
```python
if __name__ == "__main__":
    result = run_agent("What is the price of a phone with a silver discount?")
    print(result)
```

Para adicionar novas capacidades, crie mais funções com `@tool` e inclua-as na lista `tools` dentro de `run_agent`.

---

## 📝 Notas

- O `.env` e o `.venv/` estão no `.gitignore` — não devem ser versionados.
- `TAVELY_API_KEY` aparece no `.env` de exemplo, mas não é usado por `hello.py` (provavelmente de um experimento separado de busca).
- Execute `black .` e `isort .` para manter a formatação consistente antes de commitar.
