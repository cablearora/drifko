from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.llms import HuggingFaceHub
import googlesearch
import asyncio


async def process_legal_question(question):
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        huggingfacehub_api_token="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",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
        },
    )

    # Check if the question is legal-related
    classification = llm.generate([question])
    if classification['label'] == 'legal':
        # Search Google for relevant URLs
        urls = []
        for url in googlesearch.search(question, num=5, stop=5):
            urls.append(url)

        # Load content from the top 5 URLs
        url_loader = UnstructuredURLLoader()
        extracted_content = await asyncio.gather(*[url_loader.load(url) for url in urls])

        return extracted_content
    else:
        return "Not a legal-related question."
