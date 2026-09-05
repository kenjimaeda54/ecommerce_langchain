# E-commerce LangChain Agent

Um **agente de IA em Python** construído com [LangChain](https://www.langchain.com/) que atua como um assistente de compras. Ele usa o padrão de arquitetura **ReAct (Reasoning + Acting)** para consultar um catálogo de produtos e aplicar descontos por meio de ferramentas (tools), em vez de "inventar" respostas.

O projeto contém **duas implementações do mesmo agente**, resolvendo exatamente a mesma tarefa:

| Arquivo | O que é |
|---------|---------|
| `hello.py` | Implementação **manual/didática** do loop ReAct — você escreve o `for`, executa as tools e monta as mensagens na mão |
| `hello_agent.py` | Implementação **de produção** — usa `create_agent` (LangGraph) e deixa a LangChain orquestrar o loop |

> Se você quer **aprender** como um agente funciona por dentro, leia o `hello.py`. Se quer **escrever código real**, use o `hello_agent.py`.


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

## 🧩 Como chegamos a esse código (raciocínio  passo a passo)

Esta seção é um "mapa mental".

### A cadeia de problemas → soluções

| # | Problema observado | Decisão / Solução | Onde aparece no código |
|---|--------------------|-------------------|--------------------------|
| 1 | O LLM sozinho **inventa** preços (alucina) | Dar ao modelo **fontes de dados reais** → ferramentas | funções com `@tool` |
| 2 | O modelo não chama a função sozinho | **"Apresentar"** as ferramentas a ele | `llm.bind_tools(tools)` |
| 3 | O modelo só devolve a *intenção* de chamar, não o resultado | **Executar a função manualmente** | `tool_call_use.invoke(tool_call_args)` |
| 4 | O modelo não tem memória entre chamadas | **Devolver o resultado** como mensagem e reenviar o histórico | `messages.append(ToolMessage(...))` |
| 5 | Precisamos repetir até a resposta final | Criar o **loop** que intercala decisão → ação → observação | `for interaction in range(...)` |
| 6 | O modelo pode pular etapas e errar | **Guiar o raciocínio** com regras estritas | `SystemMessage` |

> 🔑 Esse encadeamento (passos 1→5) **é a própria arquitetura ReAct**. O loop foi feito à mão justamente para *ver* a engrenagem por dentro. Em produção, esse mesmo esqueleto já existe pronto na LangChain (`create_agent` + LangGraph) — é exatamente o que o `hello_agent.py` faz, e a lógica interna é idêntica.

### Lendo o código com esse mapa na cabeça

Como interpretar o `hello.py` , leia-o nesta ordem:

1. **`@tool`** → "aqui são os poderes do agente" (o que ele *pode fazer* no mundo real).
2. **`init_chat_model` + `bind_tools`** → "aqui o cérebro é ligado às ferramentas" (o que ele *sabe que pode fazer*).
3. **`SystemMessage`** → "aqui estão as regras de comportamento" (como ele *deve raciocinar*).
4. **`messages` + loop** → "aqui roda o ciclo Thought → Action → Observation" (a execução).
5. **`ToolMessage`** → "aqui o resultado da ação volta para a memória" (a observação).

> 📌 **Importante:** tudo isso acima descreve o `hello.py`, a versão **didática**. O `hello_agent.py` resolve o mesmo problema sem você escrever nenhuma dessas engrenagens — veja a próxima seção.

---

## ⚖️ Manual vs. produção: `hello.py` e `hello_agent.py`

Os dois arquivos fazem **a mesma coisa** (dado *"preço de um laptop com desconto gold"*, respondem `$750.00`), usam **as mesmas tools**, o **mesmo modelo**, o **mesmo prompt de sistema** e o **mesmo limite de 10 iterações**. A diferença está em **quem controla o loop ReAct**.

### `hello.py` — implementação manual (didática)

Você é o motor do agente. Cada passo do ReAct está explícito no código:

```python
llm_with_tools = llm.bind_tools(tools)          # 1. apresenta as tools ao modelo

for interaction in range(1, MAX_INTERACTIONS + 1):
    ai_messages = llm_with_tools.invoke(messages)   # 2. Thought: modelo decide
    tools_calls = ai_messages.tool_calls

    if not tools_calls:                             # 3. sem ação → resposta final
        return ai_messages.content

    tool_call = tools_calls[0]                      # 4. Action: executa na mão
    observation = tools_dict[tool_call["name"]].invoke(tool_call["args"])

    messages.append(ai_messages)                    # 5. Observation: devolve p/ memória
    messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
```

- ✅ Você **vê** a engrenagem: bind → invoke → executa → anexa → repete.
- ❌ Muito código de **encanamento** (plumbing) que não tem nada a ver com a regra de negócio.
- ❌ Só trata **uma** tool call por volta (`tools_calls[0]`) — se o modelo pedir duas tools de uma vez, a segunda é ignorada.
- ❌ Sem memória entre execuções: cada `run_agent()` começa do zero.
- ❌ Você reimplementa algo que o framework já faz (e melhor).

### `hello_agent.py` — implementação de produção (LangChain / LangGraph)

Você **descreve** o agente; o `create_agent` monta o grafo que executa o loop:

```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,      # substitui o SystemMessage manual
)

config: RunnableConfig = {"recursion_limit": MAX_ITERATIONS}   # trava o loop

result = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    config=config,
)
return result["messages"][-1].content
```

- ✅ **Zero loop escrito à mão** — o grafo já faz Thought → Action → Observation até não haver mais `tool_calls`.
- ✅ Executa **várias tool calls em paralelo** na mesma volta.
- ✅ **Histórico/estado** gerenciado pelo grafo (com `checkpointer` você ganha memória persistente de graça).
- ✅ **Streaming**, retomada de execução, human-in-the-loop, retries e **tracing** nativos.
- ✅ Muito menos código para errar — você só mantém tools + prompt.

### Resumo lado a lado

| Aspecto | `hello.py` (manual) | `hello_agent.py` (produção) |
|---------|---------------------|------------------------------|
| Quem roda o loop ReAct | Seu `for` | `create_agent` (grafo LangGraph) |
| `bind_tools` | Explícito (`llm.bind_tools(tools)`) | Interno ao `create_agent` |
| Execução das tools | Manual (`tools_dict[...].invoke(...)`) | Automática pelo grafo |
| `ToolMessage` (observação) | Você dá `append` na mão | O grafo anexa |
| Múltiplas tool calls por volta | ❌ só a primeira (`tools_calls[0]`) | ✅ todas, em paralelo |
| Prompt de sistema | `SystemMessage(...)` dentro de `messages` | Parâmetro `system_prompt=` |
| Limite de iterações | `MAX_INTERACTIONS` no `range()` | `recursion_limit` no `RunnableConfig` |
| Memória entre execuções | ❌ nenhuma | ✅ via `checkpointer` |
| Histórico de mensagens | Lista `messages` local | Estado do grafo (`result["messages"]`) |
| Linhas de código do loop | ~25 | 0 |
| **Serve para** | **Entender** o que um agente faz por dentro | **Escrever** agentes de verdade |

> 🔑 **A moral da história:** o `hello.py` não é "código descartável" — ele é o *mapa* do que o `create_agent` faz escondido. Depois de ler os dois, fica óbvio que a versão de produção é a mesma máquina, só que com o motor encapsulado. Em projeto real, use `hello_agent.py`.

---

## 🛠️ Stack / Tecnologias

- **Python 3.10+** — linguagem do projeto (back-end / script de terminal)
- **LangChain** (`langchain`, `langchain-core`) — orquestração do agente e ferramentas
- **LangGraph** (embutido no `langchain` 1.x) — é o motor que executa o loop ReAct criado por `create_agent` em `hello_agent.py`
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
├── hello.py          # Implementação MANUAL do loop ReAct (didática — mostra a engrenagem)
├── hello_agent.py    # Implementação de PRODUÇÃO com create_agent (LangGraph roda o loop)
├── pyproject.toml    # Dependências e metadados (gerenciado via uv)
├── uv.lock           # Lockfile de dependências
├── .env              # Variáveis de ambiente (API keys) — NÃO versionar
├── .gitignore        # Ignora .venv, .env, __pycache__
└── README.md         # Este arquivo
```

**Qual arquivo rodar?**

- `hello.py` → para **estudar** o padrão ReAct passo a passo.
- `hello_agent.py` → para **usar** o agente (é o que você levaria para produção).

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
uv run python hello.py          # versão manual (didática)
uv run python hello_agent.py    # versão de produção (create_agent)
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
# Implementação manual do loop ReAct (didática)
python hello.py

# Implementação de produção com create_agent / LangGraph
python hello_agent.py
```

Saída esperada (exemplo, **igual nos dois**):
```
The price of a laptop after a gold discount is $750.00.
```

---

## 🔧 Como funciona cada etapa (detalhado)

> As etapas abaixo descrevem o **`hello.py`** (implementação manual). No `hello_agent.py` as etapas 3–5 são abstraídas pelo `create_agent`.

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

### 5. O loop ReAct (`for interaction in range(...)`) — só no `hello.py`
A cada volta:
1. O modelo recebe todo o histórico (`messages`) e decide o próximo passo.
2. Se houver `tool_calls`, executa a ferramenta e anexa o resultado como `ToolMessage`.
3. Se **não** houver `tool_calls`, o modelo já tem a resposta → retorna o conteúdo.
4. O modelo **não tem memória própria**: o histórico é reenviado toda volta (isso é a "caderneta" do ReAct).

### 5-b. O mesmo loop, pronto (`create_agent`) — só no `hello_agent.py`
Nenhuma das etapas acima é escrita à mão. O `create_agent` devolve um grafo LangGraph com o ciclo
`modelo → executa tools → Observation → modelo` já montado; você só invoca:

```python
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
result = agent.invoke({"messages": [{"role": "user", "content": question}]},
                      config={"recursion_limit": MAX_ITERATIONS})
```

O `recursion_limit` faz o papel do `MAX_INTERACTIONS` e o histórico volta pronto em `result["messages"]`.

---

## 📊 Observabilidade com LangSmith

Com `LANGSMITH_TRACING=true`, cada execução é registrada no [LangSmith](https://smith.langchain.com/). O decorator `traceable` marca o início do trace, permitindo inspecionar cada Thought/Action/Observation no dashboard:

```python
@traceable(name="LangChain Agent Loop")   # hello.py
def run_agent(question: str):
```

```python
@traceable(name="LangChain create_agent") # hello_agent.py
def run_agent(question: str):
```

Assim os dois traces aparecem separados no LangSmith e você pode comparar a execução manual com a do grafo.

---

## 🧪 Modificando o agente

Para testar outras perguntas, altere a chamada no final do arquivo (`hello.py` **ou** `hello_agent.py`, ambos têm o mesmo bloco):

```python
if __name__ == "__main__":
    result = run_agent("What is the price of a phone with a silver discount?")
    print(result)
```

Para adicionar novas capacidades, crie mais funções com `@tool` e inclua-as na lista `tools` dentro de `run_agent` — o procedimento é idêntico nos dois arquivos (é a única parte que você continua escrevendo à mão na versão de produção).

---

## 📝 Notas

- O `.env` e o `.venv/` estão no `.gitignore` — não devem ser versionados.
- `TAVELY_API_KEY` aparece no `.env` de exemplo, mas não é usado por `hello.py` (provavelmente de um experimento separado de busca).
- Execute `black .` e `isort .` para manter a formatação consistente antes de commitar.
