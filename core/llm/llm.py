# -*- coding: utf-8 -*-
"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, and Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""

import os
import openai

_SESSION_STATE = {"msg_history": []}


def _cache_resource(*args, **kwargs):
    def decorator(func):
        return func

    return decorator
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
# from llama_index.core.llms import ChatMessage, MessageRole
# from llama_index.core.node_parser import SimpleNodeParser
# from llama_index.core.indices.loading import load_index_from_storage
# from llama_index.llms.openai import OpenAI


def get_llm_runtime_config(preferred_model=None):
    """
        Resolve runtime LLM configuration from environment variables only.
        No secrets or proxy endpoints are hardcoded in source code.
    """

    model = (
        preferred_model
        or os.environ.get("PE_MAS_LLM_MODEL")
        or "gpt-4o-mini"
    )
    api_key = os.environ.get("PE_MAS_LLM_CREDENTIAL")
    api_base = (
        os.environ.get("PE_MAS_LLM_ENDPOINT")
        or ""
    )
    return {
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
    }


def openai_init(openai_model=None, api_key=None, api_url=None, require_key=False):
    """
        initialize OpenAI, including api_key and api_url
    """    

    runtime = get_llm_runtime_config(preferred_model=openai_model)
    if api_key is None:
        api_key = runtime["api_key"]
    if api_url is None:
        api_url = runtime["api_base"]

    if require_key and not api_key:
        raise RuntimeError(
            "Model provider credential is not configured. Set PE_MAS_LLM_CREDENTIAL locally before running PE-MAS."
        )

    if not api_key:
        return None

    openai.api_key = api_key

    client_kwargs = {"api_key": openai.api_key}
    if api_url:
        openai.base_url = api_url
        client_kwargs["base_url"] = api_url

    client = openai.OpenAI(**client_kwargs)
    
    return client

def get_agent_reasoning(agent_role: str, task: str, context_data: dict, history_trace: list = None) -> list:
    """
    Generates verbose, SOTA-level agent reasoning logs (S2R).
    Simulates: ReAct (Reasoning + Acting) + Self-Reflection.
    Now supports Project Context (Memory) to avoid loops.
    """
    runtime = get_llm_runtime_config(preferred_model=os.environ.get("PE_MAS_REASONING_MODEL") or "gpt-4o")
    client = openai_init(
        openai_model=runtime["model"],
        api_key=runtime["api_key"],
        api_url=runtime["api_base"],
        require_key=True,
    )
    
    # Compress context
    import json
    context_str = json.dumps(context_data, default=str)[:3000]
    
    # Format History Trace for Context
    history_block = ""
    if history_trace and len(history_trace) > 0:
        import json
        trace_strs = []
        for item in history_trace[-5:]:
            if isinstance(item, dict):
                try:
                    trace_strs.append(json.dumps(item))
                except (TypeError, ValueError):
                    trace_strs.append(str(item))
            else:
                trace_strs.append(str(item))
                
        history_block = f"\n\n    PREVIOUS ATTEMPTS (MEMORY):\n    " + "\n    ".join(trace_strs) + "\n    (Review this memory to avoid repeating failed strategies!)"
    
    system_prompt = f"""You are an advanced autonomous AI Agent (Role: {agent_role}) operating in a Multi-Agent System (MAS).
    Your cognitive architecture mimics human expert reasoning (System 2).
    
    OBJECTIVE:
    Deeply analyze the task before execution. Do not be brief. Show your entire thought process, including:
    1.  **Context Analysis**: Decode the physical and electrical constraints.
    2.  **Tool Selection**: Explicitly mention tools you will use.
    3.  **Risk Assessment**: Predict failure points.
    4.  **Strategy Formulation**: Decide on optimal parameters.{history_block}
    
    OUTPUT FORMAT:
    Return 8-12 detailed log lines. Use these tags:
    [OBSERVATION] - What you see in the input data.
    [KNOWLEDGE] - Relevant physics/domain rules.
    [THOUGHT] - Your internal deduction or trade-off analysis.
    [TOOL] - The specific tool you are activating.
    [PLAN] - The step-by-step execution path.
    [CRITIQUE] - Self-correction or safety check.

    CRITICAL FORMATTING RULE:
    **Do NOT use newlines for the content of a tag.**
    Correct: "[PLAN] Step 1: Calculate parameters. Step 2: Select MOSFET."

    TONE:
    Highly technical, precise, and authoritative.
    """
    
    user_prompt = f"Current Task: {task}\nOperational Context: {context_str}"
    
    try:
        response = client.chat.completions.create(
            model=runtime["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4, # Slightly higher for more creative reasoning
            max_tokens=2000   # Allow longer generation for complete thought chains
        )
        content = response.choices[0].message.content
        
        # Robust Parsing: Handle multi-line responses by merging them to previous tag
        processed_lines = []
        raw_lines = content.split('\n')
        
        current_tag_line = ""
        
        for line in raw_lines:
            line = line.strip()
            if not line: continue
            
            if line.startswith('['):
                # If we have a previous tag pending, save it
                if current_tag_line:
                    processed_lines.append(current_tag_line)
                current_tag_line = line
            else:
                # If it's a continuation line (no tag), append to previous
                if current_tag_line:
                    current_tag_line += " " + line
                else:
                    # Case where output starts without a tag (unlikely but possible)
                    current_tag_line = f"[THOUGHT] {line}"
        
        # Append the last one
        if current_tag_line:
            processed_lines.append(current_tag_line)
            
        return processed_lines

    except Exception as e:
        return [f"[ERROR] Cognitive Engine Offline: {str(e)}", "[FALLBACK] Executing standard protocol."]


# Specialized multi-agents to handle different tasks with Retrieval Augmented Generation (RAG)
@_cache_resource(show_spinner=False)
def rag_load(database_folder, llm_model, temperature=None, 
             chunk_size=None, system_prompt=None, recreate_index=False):
    """
        This function is the retrieval-augmented generation (RAG) for LLM
    """
    
    if chunk_size is None: chunk_size = 1024
    if temperature is None: temperature = 0.0
    
    # Lazy Import to prevent Torch/SSL conflicts
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.indices.loading import load_index_from_storage
    from llama_index.llms.openai import OpenAI
    
    # path for storing/loading index to avoid repetitive vectorizing
    index_storage_path = database_folder+"/saved_index"
    
    if os.path.exists(index_storage_path+"/docstore.json") and (not recreate_index):
        storage_context = StorageContext.from_defaults(persist_dir=index_storage_path)
        index = load_index_from_storage(storage_context)
    else:
        llm = OpenAI(model=llm_model, temperature=temperature,
                     system_prompt=system_prompt)
        docs = SimpleDirectoryReader(database_folder).load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        nodes = node_parser.get_nodes_from_documents(docs)
        
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex(nodes, storage_context=storage_context, llm_predictor=llm)
        index.storage_context.persist(persist_dir=index_storage_path)
    from llama_index.core.llms import ChatMessage, MessageRole
    return index


def get_msg_history():
    """
        get the message history, mainly used for chat engines (not query engines)
    """
    from llama_index.core.llms import ChatMessage, MessageRole

    msg_history = [ChatMessage(role=MessageRole.USER
                               if msg["role"] == "user"
                               else MessageRole.ASSISTANT,
                               content=msg["content"])
                   for msg in _SESSION_STATE.get("msg_history", [])]
    return msg_history


def save_msg_history(prompt: str, response: str, plot=None, history_len=None):
    """
        get the message history, mainly used for chat engines (not query engines)
    """
    
    assert (history_len is None) or (history_len >= 0), "history_len should be None or >= 0."
    if history_len == 0:
        _SESSION_STATE["msg_history"] = []
        return 
    
    if _SESSION_STATE.get("msg_history") is None:
        _SESSION_STATE["msg_history"] = []
    # Store this round of chat messages (only those related to power converter design tasks)
    _SESSION_STATE["msg_history"].append({"role": "user", "content": prompt}) # user prompt to memory
    if plot is None:
        _SESSION_STATE["msg_history"].append({"role": "assistant", 
                                             "content": response}) # pe-gpt response to memory
    else:
        _SESSION_STATE["msg_history"].append({"role": "assistant", "content": response, 
                                             "images": plot}) # pe-gpt response and plot to memory
    if history_len is not None:
        _SESSION_STATE["msg_history"] = _SESSION_STATE["msg_history"][-history_len:]
    
    
    
    
