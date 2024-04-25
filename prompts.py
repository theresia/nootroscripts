prompts = {
    # default prompt, for extracting Questions in the text and generate one-sentence answers from
    'QnAs': "list all the questions asked and a three-sentence answer to each (include all examples of concrete situations and stories shared in the answer)",
    'note': "You are an expert at making factual, succinct, and detailed notes from transcripts. " \
            "You will rewrite the transcript provided into notes. Do not summarize and keep every information. ",
    'summary': "Summarise the text in first-person as if the speaker has produced a short version of the original conversation. " \
              "Include the anecdotes, key points, arguments, and actionable takeaways. " \
              "Inject some narrative and bullet points as appropriate into your summary so the summary can be easily read. " \
              "Please use simple language and don't repeat points you have previously made.",
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
}