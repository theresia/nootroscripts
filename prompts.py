system_prompts = {
    'default': "Answer in your own style", # apparently system prompts, when specified, can't be an empty string
    'doubtful_stylistic_british': """
First and foremost you have to be able to self reflect and doubt your own immediate responses. Don't hesitate to ask questions to clarify my own goal based on the prompt and suggest other prompts that might improve the clarity of thought.
The most important thing is to be explicit and elaborate on your reasoning. I know you are a large language model but be confident and have conviction in your final recommendation.
Always suggest an interesting alternative frame of the issue using interdisciplinary angle. I don't mind long responses, as long as they are coherent, systematic, and thorough.
Please answer in 9th grade reading level and use British English spelling.
When you're summarising, paraphrasing, rewriting, and transforming text, as much as possible keep the rhetorical devices and figures of speech of the original text. I like the way Seth Godin and Elizabeth Gilbert writes and speaks.
It's very important that you get this right.
    """,
}

user_prompts = {
    # default prompt, for extracting Questions in the text and generate one-sentence answers from
    'QnAs': "list all the questions asked and a three-sentence answer to each (include all examples of concrete situations and stories shared in the answer)",
    'note': "You are an expert at making factual, succinct, and detailed notes from transcripts. " \
            "You will rewrite the transcript provided into notes. Do not summarize and keep every information. " \
            "Don't repeat points you have previously made, and don't include anything that's not in the text.",
    'summary': "Summarise the text in first-person as if the speaker has produced a short version of the original conversation. " \
               "Include the anecdotes, key points, arguments, and actionable takeaways. " \
               "Inject some narrative and bullet points as appropriate into your summary so the summary can be easily read. " \
               "Don't repeat points you have previously made, and don't include anything that's not in the text.",
    'translation': "You are a translator who handles English to Indonesian and vice versa. " \
                   "Please produce an accurate translation of the transcribed text. "\
                   "If the text is in English, then translate to all languages you know. " \
                   "If the text is non English, please translate it to English",
    # perhaps same as tagging. this works but not meaningful enough
    'topix': "Extract the 3 topics / themes / concepts that you see. Please use Wikipedia's concept taxonomy for it.",
    'tag': "what are some hashtags appropriate for this?",
    'definition': "list of all definitions made in the discussion",
    # still very.... robotic? idk how to describe it. it's nice, but not... engaging? like first grader. lack flare, personality, hooks?
    'thread': "Rewrite as a Twitter thread of 6 tweets or less. Be as granular as possible. " \
              "Speak in first person, use informal, conversational, and witty style. Write for the ears rather than for the eyes. " \
              "Introduce the essay with the surprising actionable takeaway and then go over the main arguments made in the essay. ",
    # for Instagram post mostly, if not specifically
    'tp': "Please create 2 talking points I can use for a video script through the lens of human nature based on how different arguments made in comments relate to the content and argument of the main post." \
          "Speak in first person narrative. Use informal, conversational, and witty style. Write for the ears rather than for the eyes. " \
          "Use active voice. Avoid passive voice as much as possible.",
    'cbb': "Here's a news article. please turn the title of the article into a question and find the answer to the question in the text provided",
    'density': "Based on the density of idea, conciseness, and amount of fillers in the text, please rate it on a scale of 1 to 5 (5 being the least amount of summarisation needed). Then summarise accordingly. Adjust the granularity based on your assessment. If you think it is sparse then be less granular and meticulous. If you think it is dense then be more granular and meticulous. Use bulletpoints to highlight the arguments, concepts, and recommendations if any",
    'distinctions': "give me a list of distinctions made where different nuanced term for a similar idea are contrasted",
    'misconceptions': "a list of misconceptions argued against by the speakers",
    'ada': "list of assertions, distinctions, and arguments made",
}
