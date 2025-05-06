import pandas as pd
import string

# In this file, I want to add a collection of prompts, that we can apply to the csv. 

train_data = pd.read_csv("training.csv")

# diese funktion enthält unsere Prompt Beispiele und nimmt den Typen den wir haben wollen und eine review, und applied den prompt darauf. 
# das ganze muss jetzt auf jede Zeile des csvs angewandt werden und dann in ein csv gespeichert werden. 
def applyPrompts(prompt_type: string, review:string): 
    prompts = []

    prompt_templates = [
        #question: 
        [f"What is the sentiment of this review? [{review}] The sentiment is [SENTIMENT].",
        f"Is the sentiment of this review positive, neutral, or negative? [{review}] The sentiment is [[SENTIMENT]]."], 
        
        #completion: 
        [f"[{review}]. The review expresses a [[SENTIMENT]] opinion of the experience.", 
        f"[{review}]. The user has a [[SENTIMENT]] opinion of the experience.", 
        f"[{review}]. The reviewer’s impression of the experience is [[SENTIMENT]].", 
        f"[{review}]. The reviewer’s overall assessment of the experience is [[SENTIMENT]].", 
        f"[{review}]. The sentiment expressed in the review is [[SENTIMENT]].", 
        f"[{review}]. The reviewer’s reaction to the experience is [[SENTIMENT]]."], 
        
        #dialogue: 
        [f"User: What is the sentiment of this review? [{review}] Agent: The sentiment is [[MASK]].", 
        f"User: Can you analyze the sentiment in this review? [{review}] Agent: It is a [[MASK]] review.", 
        f"User: Please determine the sentiment expressed here: [{review}] Agent: Sentiment: [[MASK]].", 
        f"User: Here's a review: [{review}] Agent: The reviewer felt [[MASK]] about their experience.", 
        f"User: Based on the following review, classify the sentiment. [{review}] Agent: Classification result: [[MASK]]."], 
    
        #contextual: 
        [f"The following review expresses a [[MASK]] sentiment: [{review}]", 
        f"This is a [[MASK]] review from the Tripadvisor dataset: [{review}]", 
        f"Analyze the sentiment of the review below. The sentiment is [[MASK]]. [{review}]", 
        f"Sentiment classification task. Input: [{review}] Label: [[MASK]]", 
        f"Given the review below, identify the sentiment expressed by the user. [{review}] Answer: [[MASK]"]
    
        #"chain of thought": [] #TODO
    ]
    match prompt_type: 
        case "question": 
            return  prompt_templates[0]
        case "completion": 
            return prompt_templates[1]
        case "dialogue": 
            return prompt_templates[2]
        case "contextual": 
            return prompt_templates[3]

    return prompt_templates

prompts_ = []
labels = []
for _, row in train_data.iterrows():
    # change the first argument to "" , or any string unequal to one of the categories to apply all prompt templates. 
    prompts = applyPrompts("dialogue",row["sentence"])
    for prompt in prompts:
        prompts_.append(prompt)
        labels.append(row["label"])

prompt_data = pd.DataFrame({"prompt": prompts_, "label": labels})  
#prompt_data.to_csv("contextual_prompts_training.csv", index=False)
#prompt_data.to_csv("question_prompts_training.csv", index=False)
#prompt_data.to_csv("completion_prompts_training.csv", index=False)
#prompt_data.to_csv("dialogue_prompts_training.csv", index=False)



