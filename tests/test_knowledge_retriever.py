from src.knowledge_retriever import retrieve_answer


def test_knowledge_retriever():
    result = retrieve_answer("Who created Python?")
    assert result is not None
    assert "Python" in result["answer"]
