# ES Mapping,
# Title is supported prefix search and full text search
# Content is supported full text search
mapping = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "title": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    },
                    "suggest": {
                        "type": "completion"
                    }
                }
            },
            "wikitext": {"type": "text"},
            "content": {"type": "text"},
        }
    }
}
