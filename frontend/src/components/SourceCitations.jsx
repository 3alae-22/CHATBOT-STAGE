import { useState } from 'react';

const isArabic = (text) => /[\u0600-\u06FF]/.test(text);

export default function SourceCitations({ sources }) {
    const [openIndex, setOpenIndex] = useState(null);

    if (!sources || sources.length === 0) return null;

    return (
        <div className="annexes">
            <div className="annexes-title">Annexes — sources citées</div>
            {sources.map((source, i) => (
                <div className="annexe-item" key={i}>
                    <div className="annexe-header" onClick={() => setOpenIndex(openIndex === i ? null : i)}>
                        <span className="annexe-number">Annexe {i + 1}</span>
                        <span className="annexe-meta">
                            {source.pdf_name} <span className="page">— page {source.page_num}</span>
                        </span>
                        <span className="annexe-toggle">{openIndex === i ? 'masquer' : "voir l'extrait"}</span>
                    </div>
                    {openIndex === i && (
                        <div className="annexe-extract" dir={isArabic(source.chunk_text) ? 'rtl' : 'ltr'}>
                            {source.chunk_text}
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}