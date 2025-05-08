import lmql

query_string = """
"Q: What is the sentiment of the following review: ```The food was very good.```?\\n"
"A: Let's think step by step. [ANALYSIS]. Therefore, the sentiment is [SENTIMENT]" where (len(TOKENS(ANALYSIS)) < 200) and STOPS_AT(ANALYSIS, '\\n') \
    and (SENTIMENT in ['positive', 'negative', 'neutral'])
"""

model = lmql.model("llama.cpp:/Users/deckk/CIL/CILProject2025/zephyr-7b-beta.Q4_K_M.gguf",
       tokenizer = 'HuggingFaceH4/zephyr-7b-beta')

out = lmql.run_sync(query_string, model = model).variables['SENTIMENT']

print(out)