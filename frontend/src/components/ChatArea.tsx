import React, { useEffect, useRef, memo } from 'react';
import { Scale } from 'lucide-react';
import type { Message, CaseStatus, LoadingStatus } from '../App';
import ReactMarkdown from 'react-markdown';

type ChatAreaProps = {
  messages: Message[];
  loadingStatus: LoadingStatus;
  caseStatus: CaseStatus;
};

const MessageRow = memo(({ msg }: { msg: Message }) => {
  return (
    <div className={`message-row ${msg.role}`}>
      {msg.role === 'assistant' ? (
        <div className="assistant-message-wrapper">
          <div className="assistant-avatar">
            <Scale size={20} color="#000" />
          </div>
          <div className="message-bubble assistant glass-panel">
            <div className="markdown-content">
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          </div>
        </div>
      ) : (
        <div className="message-bubble user glass-user-panel">
          {msg.content}
        </div>
      )}
    </div>
  );
});

export const ChatArea: React.FC<ChatAreaProps> = memo(({ messages, loadingStatus, caseStatus }) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loadingStatus.active]);

  return (
    <div className="chat-container">
      <div className="chat-content-width">
        {/* Show document summary when no messages yet */}
        {caseStatus.hasCase && caseStatus.summary && messages.length === 0 && (
          <div className="message-row assistant">
            <div className="assistant-message-wrapper">
              <div className="assistant-avatar">
                <Scale size={20} color="#000" />
              </div>
              <div className="message-bubble assistant glass-panel">
                <div style={{ color: 'var(--color-accent-gold)', fontWeight: 500, marginBottom: '8px' }}>
                  📄 Document Analysis Complete
                </div>
                <div className="markdown-content">
                  <ReactMarkdown>{caseStatus.summary}</ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <MessageRow key={msg.id} msg={msg} />
        ))}

        {/* Rich loading indicator with phase text */}
        {loadingStatus.active && (
          <div className="message-row assistant">
            <div className="assistant-message-wrapper">
              <div className="assistant-avatar">
                <Scale size={20} color="#000" />
              </div>
              <div className="message-bubble assistant glass-panel loading-bubble">
                <div className="loading-phase-text">{loadingStatus.phase}</div>
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
});
