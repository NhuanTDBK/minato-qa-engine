from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document

basic_search_retriever_prompt = """
You will be given a conversation below and a follow up question. You need to rephrase the follow-up question if needed so it is a standalone question that can be used by the LLM to search the web for information.
If it is a writing task or a simple hi, hello rather than a question, you need to return `not_needed` as the response. Just output a string only, no furthur explain!

Example:
1. Follow up question: What is the capital of France?
Rephrased: Capital of france

2. Follow up question: What is the population of New York City?
Rephrased: Population of New York City

3. Follow up question: What is Docker?
Rephrased: What is Docker

Conversation:
{chat_history}

Follow up question: {query}
Rephrased question:
"""

basic_web_search_response_prompt = """
    You are Naruto QA, an AI model who is comic/manga expert at searching the web and answering user's queries.
    Given this query: {query}
    Generate a response that is informative and relevant to the user's query based on provided context (the context consits of search results containg a brief description of the content of that page).
    You must use this context to answer the user's query in the best way possible. Use an unbaised and journalistic tone in your response. Do not repeat the text.
    You must not tell the user to open any link or visit any website to get the answer. You must provide the answer in the response itself. If the user asks for links you can provide them.
    Your responses should be medium to long in length be informative and relevant to the user's query. You can use markdowns to format your response. You should use bullet points to list the information. Make sure the answer is not short and is informative.
    You have to cite the answer using [number] notation. You must cite the sentences with their relevent context number. You must cite each and every part of the answer so the user can know where the information is coming from.
    Place these citations at the end of that particular sentence. You can cite the same sentence multiple times if it is relevant to the user's query like [number1][number2].
    However you do not need to cite it using the same number. You can use different numbers to cite the same sentence multiple times. The number refers to the number of the search result (passed in the context) used to generate that part of the answer.

    Aything inside the following \`context\` HTML block provided below is for your knowledge returned by the search engine and is not shared by the user. You have to answer question on the basis of it and cite the relevant information from it but you do not have to 
    talk about the context in your response. 

    <context>
    {context}
    </context>

    If you think there's nothing relevant in the search results, you can say that 'Hmm, sorry I could not find any relevant information on this topic. Would you like me to search again or ask something else?'.
    Anything between the \`context\` is retrieved from a search engine and is not a part of the conversation with the user
"""


def refine_query(llm, query, chat_history=None):
    prompt = basic_search_retriever_prompt.format(
        chat_history=chat_history, query=query
    )
    return llm.invoke(prompt)


def summarize_websearch_conversation(
    llm, query, context, chunk_size: int = 2048, stream: bool = False
):
    prompt = basic_web_search_response_prompt.format(context=context, query=query)
    prompt = Document(page_content=prompt)
    # spliter = TokenTextSplitter(chunk_size=chunk_size)
    # doc = spliter.create_documents([prompt])
    # print(2)
    # split_docs = spliter.split_documents(doc)
    # prompt_template = split_docs[0]
    print("Content: ", prompt.page_content)
    if not stream:
        return iter([llm.invoke(prompt.page_content)])

    return llm.stream(prompt.page_content)
