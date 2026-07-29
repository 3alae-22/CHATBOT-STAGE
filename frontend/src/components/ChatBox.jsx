import { useState, useRef, useEffect } from 'react';
import SourceCitations from './SourceCitations';

const isArabic = (text) => /[\u0600-\u06FF]/.test(text);
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function ChatBox() {
    const [turns, setTurns] = useState([]);
    const [question, setQuestion] = useState('');
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [turns]);

    const askQuestion = async () => {
        const q = question.trim();
        if (!q) return;

        setQuestion('');
        setTurns((prev) => [...prev, { question: q, loading: true }]);

        try {
            const res = await fetch(`${API_URL}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: q, k: 5 }),
            });
            if (!res.ok) throw new Error(`Erreur ${res.status}`);
            const data = await res.json();

            setTurns((prev) =>
                prev.map((t, i) => (i === prev.length - 1 ? { ...t, response: data, loading: false } : t))
            );
        } catch (err) {
            setTurns((prev) =>
                prev.map((t, i) =>
                    i === prev.length - 1
                        ? {
                            ...t,
                            loading: false,
                            response: {
                                answer: "Une erreur s'est produite. Vérifiez que le service est disponible.",
                                sources: [],
                                abstained: true,
                            },
                        }
                        : t
                )
            );
        }
    };

    return (
        <div className="app-shell">
            <header className="letterhead">
                <div className="letterhead-emblem">RP</div>
                <div className="letterhead-text">
                    <h1>Assistant Documentaire</h1>
                    <p>Province - Interrogation des textes administratifs</p>
                </div>
            </header>

            <div className="conversation">
                {turns.length === 0 && (
                    <div className="empty-state">
                        <p>
                            Posez une question sur les textes réglementaires et administratifs de la
                            province, en français ou en arabe.
                        </p>
                    </div>
                )}

                {turns.map((turn, i) => (
                    <div key={i}>
                        <div className="question-bubble" dir={isArabic(turn.question) ? 'rtl' : 'ltr'}>
                            {turn.question}
                        </div>

                        {turn.loading && (
                            <div className="answer-card" style={{ marginTop: 12 }}>
                                <div className="eyebrow">Réponse</div>
                                <div className="answer-text loading-dots">Recherche dans les documents</div>
                            </div>
                        )}

                        {turn.response && (
                            <div
                                className={`answer-card ${turn.response.abstained ? 'abstained' : ''}`}
                                style={{ marginTop: 12 }}
                            >
                                {turn.response.abstained && <div className="stamp">Dossier non trouvé</div>}
                                <div className="eyebrow">
                                    <span>{turn.response.abstained ? 'Sans réponse' : 'Réponse officielle'}</span>
                                </div>
                                <div className="answer-text" dir={isArabic(turn.response.answer) ? 'rtl' : 'ltr'}>
                                    {turn.response.answer}
                                </div>
                                <SourceCitations sources={turn.response.sources} />
                            </div>
                        )}
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>

            <div className="input-bar">
                <input
                    type="text"
                    placeholder="Poser une question..."
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && askQuestion()}
                    dir={isArabic(question) ? 'rtl' : 'ltr'}
                />
                <button onClick={askQuestion} disabled={!question.trim()}>
                    Envoyer
                </button>
            </div>
        </div>
    );
}