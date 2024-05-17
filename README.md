# Minato RAG (Rasengen) Engine
# For Wibu
This site is for Naruto fanbase ask me anything with LLM-powered
- Search AMA
- News
- Stats
- Affiliate

# Database
- Page
    + id: a string, contains title
    + type: a string , which represents entity
    + content: a string
- Image
    + id: a string
    + page_id: a string reference to page
- Relation
    + from_id: 
    + to_id
    + relation

# UI Search
- First page: a textbox
- When searching
    + Showing related pages from wikifandom, google search and youtube
    + Show RAG-content
