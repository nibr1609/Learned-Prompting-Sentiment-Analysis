import lmql

query_string = """
"Q: What is the sentiment of the following review: ```The food was very good.```?\\n"
"A: [SENTIMENT]" where (len(TOKENS(SENTIMENT)) < 200)
"""

print(lmql.run_sync(
    query_string, 
    model = lmql.model("llama.cpp:/Users/deckk/CIL/CILProject2025/zephyr-7b-beta.Q4_K_M.gguf", 
        tokenizer = 'HuggingFaceH4/zephyr-7b-beta')).variables['SENTIMENT'])