import lmql

lmql.serve("llama.cpp:/Users/deckk/CIL/CILProject2025/zephyr-7b-beta.Q4_K_M.gguf", port=8080, trust_remote_code=True, n_ctx=4096)