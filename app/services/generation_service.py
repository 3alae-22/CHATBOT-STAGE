from google.genai import types

from app.services.retrieval_service import RetrievalService


INSTRUCTIONS = '''
You are an assistant that answers administrative questions based solely on excerpts from official Moroccan legal texts.

STRICT RULES:
1. Answer ONLY using the excerpts provided below. Do not use any external knowledge, even if you believe you know the answer.
2. If the excerpts do not contain the information needed to answer, respond exactly: "Je ne trouve pas cette information dans les documents fournis." Do not guess, extrapolate, or fill gaps with general legal reasoning.
3. Always cite the source of each claim in parentheses, using the format (Source X, page Y).
4. If multiple excerpts contradict each other or are ambiguous, explicitly point this out rather than resolving it yourself.
5. Answer in the same language as the question (French or Arabic). If the question is in Arabic but the relevant excerpts are in French (or vice versa), translate the source content faithfully and indicate that you are translating.
6. Do not provide legal advice or personal interpretation of the law — rephrase and cite what the text says, nothing more.
'''

USER_PROMPT_TEMPLATE = '''
Question:
{question}

Context:
{context}
'''


class GenerationService:
    def __init__(
        self,
        llm_client,
        retrieval_service: RetrievalService,
        model: str = 'gemini-2.5-flash',
        instruction: str = INSTRUCTIONS,
        prompt_template: str = USER_PROMPT_TEMPLATE,
    ):
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client
        self.model = model
        self.instruction = instruction
        self.prompt_template = prompt_template

    def build_context(self, search_results) -> str:
        lines = []
        for i, row in enumerate(search_results, start=1):
            _, pdf_name, page_num, chunk_text, distance = row
            lines.append(f"[Source {i} — {pdf_name}, page {page_num}]")
            lines.append(chunk_text.strip())
            lines.append("")
        return "\n".join(lines).strip()

    def build_prompt(self, question: str, search_results) -> str:
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(question=question, context=context)
        return prompt.strip()

    def generate(self, prompt: str) -> str:

        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=self.instruction),
        )
        return response.text

    def ask(self, question: str, k: int = 5) -> dict:

        search_results = self.retrieval_service.search(question, k=k)
        prompt = self.build_prompt(question, search_results)
        answer = self.generate(prompt)

        sources = [
            {"pdf_name": row[1], "page_num": row[2], "chunk_text": row[3]}
            for row in search_results
        ]
        return {"answer": answer, "sources": sources}


    
