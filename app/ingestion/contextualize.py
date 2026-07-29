import ollama


DOC_SUMMARY_SYSTEM_PROMPT = """You are given the first page(s) of an official
administrative/legal document. Write a short summary (2-3 sentences) describing
its general subject and structure (e.g. which law/decree it is, what topics it covers).

CRITICAL RULES:
- You MUST write the summary in the SAME language as the document text provided
  below. If the document is in French, respond ONLY in French. If the document
  is in Arabic, respond ONLY in Arabic.
- ONLY state facts that are explicitly present in the text provided. Do NOT
  infer, guess, or add any country, institution, date, or detail that is not
  literally written in the excerpt. If unsure, stay vague rather than invent.
- Output ONLY the summary, nothing else.
- Keep it under 60 words."""

CONTEXT_SYSTEM_PROMPT = """You are given a document summary and a short excerpt
(chunk) taken from that document. Write a single short sentence that situates
this excerpt within the document.

CRITICAL RULES:
- You MUST write the sentence in the SAME language as the excerpt provided
  below (indicated explicitly). Never respond in a different language.
- ONLY state facts explicitly present in the summary or excerpt. Do NOT invent
  any detail (country, date, institution, article number) not literally present.
- Output ONLY the context sentence, nothing else.
- Do not repeat the excerpt itself.
- Keep it under 30 words."""


def generate_doc_summary(first_pages_text: str, lang: str, model: str = "qwen2.5:7b") -> str:
    lang_name = "French" if lang == "fr" else "Arabic"
    prompt = f"[Respond in {lang_name} only]\n\n{first_pages_text}"
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": DOC_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"].strip()


def generate_chunk_context(chunk: str, doc_summary: str, lang: str, model: str = "qwen2.5:7b") -> str:
    lang_name = "French" if lang == "fr" else "Arabic"
    prompt = f"[Respond in {lang_name} only]\n\nDocument summary:\n{doc_summary}\n\nExcerpt:\n{chunk}"
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"].strip()


def contextualize_chunk(chunk: str, doc_summary: str, lang: str, model: str = "qwen2.5:7b") -> str:
    context = generate_chunk_context(chunk, doc_summary, lang, model=model)
    return f"{context}\n\n{chunk}"