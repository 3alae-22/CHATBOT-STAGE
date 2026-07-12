import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.title("Assistant administratif")

question = st.text_input("Pose ta question :")

if st.button("Envoyer"):
    if not question.strip():
        st.warning("Écris une question avant d'envoyer.")
    else:
        with st.spinner("Recherche en cours..."):
            try:
                response = requests.post(API_URL, json={"question": question, "k": 5})
                response.raise_for_status()
                data = response.json()

                st.subheader("Réponse")
                st.write(data["answer"])

                st.subheader("Sources")
                for source in data["sources"]:
                    st.markdown(
                        f"**{source['pdf_name']}** — page {source['page_num']}"
                    )
                    with st.expander("Voir l'extrait"):
                        st.write(source["chunk_text"])

            except requests.exceptions.ConnectionError:
                st.error("Impossible de joindre l'API. Vérifie que FastAPI tourne (uvicorn) sur le port 8000.")
            except requests.exceptions.HTTPError as e:
                st.error(f"Erreur de l'API : {e}")