import React from 'react';
import { Scale, Clock, MessageSquare, ShieldCheck, Trash2, Plus } from 'lucide-react';
import type { CaseStatus, Conversation } from '../App';

type SidebarProps = {
  language: string;
  setLanguage: (lang: string) => void;
  onClearDocument: () => void;
  caseStatus: CaseStatus;
  conversations: Conversation[];
  currentConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
};

const SUPPORTED_LANGUAGES = [
  "Auto (Match My Input)",
  "English", "Hindi", "Telugu", "Tamil", 
  "Kannada", "Malayalam", "Marathi", 
  "Bengali", "Gujarati", "Urdu", "Arabic", "Punjabi"
];

const formatTimestamp = (ts: number) => {
  const date = new Date(ts);
  const now = new Date();
  const isToday = date.getDate() === now.getDate() && date.getMonth() === now.getMonth() && date.getFullYear() === now.getFullYear();
  const isYesterday = new Date(now.setDate(now.getDate() - 1)).getDate() === date.getDate() && date.getMonth() === now.getMonth() && date.getFullYear() === now.getFullYear();
  
  if (isToday) {
    return `Today at ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  } else if (isYesterday) {
    return 'Yesterday';
  } else {
    return date.toLocaleDateString();
  }
};

export const Sidebar: React.FC<SidebarProps> = ({ 
  language, 
  setLanguage, 
  onClearDocument,
  caseStatus,
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewChat
}) => {
  return (
    <aside className="sidebar">
      <div className="brand-section">
        <div className="brand-icon">
          <Scale size={24} />
        </div>
        <div className="brand-text">
          <h1>NomoSys</h1>
          <p>AI Legal Chatbot</p>
        </div>
      </div>

      <div className="sidebar-section">
        <h3>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ width: '16px', height: '16px', display: 'inline-block' }}>🌐</span> Language
          </span>
        </h3>
        <select 
          className="language-select" 
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
        >
          {SUPPORTED_LANGUAGES.map(lang => (
            <option key={lang} value={lang}>{lang}</option>
          ))}
        </select>
      </div>

      <div className="sidebar-section" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <h3 style={{ alignItems: 'center' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Clock size={16} /> Memory History
          </span>
          <button 
            onClick={onNewChat}
            style={{ 
              background: 'none', border: 'none', color: 'var(--color-accent-gold)', 
              fontSize: '0.8rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px'
            }}
          >
            <Plus size={14} /> New Chat
          </button>
        </h3>
        <div className="history-list">
          {conversations.length === 0 ? (
            <div style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', textAlign: 'center', padding: '1rem' }}>
              No previous conversations
            </div>
          ) : (
            conversations.map(conv => (
              <div 
                key={conv.id} 
                className={`history-item ${conv.id === currentConversationId ? 'active' : ''}`}
                onClick={() => onSelectConversation(conv.id)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                  <MessageSquare size={14} color={conv.id === currentConversationId ? "var(--color-accent-gold)" : "currentColor"} />
                  <span className="history-item-title">{conv.title}</span>
                </div>
                <span className="history-item-time">{formatTimestamp(conv.timestamp).split(' at ')[0]}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {caseStatus.hasCase && (
        <div className="sidebar-section" style={{ marginTop: 'auto', marginBottom: '1rem' }}>
          <h3>Loaded Document</h3>
          <div className="document-card" style={{ padding: '0.75rem', marginBottom: 0, backgroundColor: 'rgba(212,175,55,0.1)' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--color-text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              📄 {caseStatus.fileName || 'Uploaded Document'}
            </div>
            <button 
              onClick={onClearDocument}
              style={{
                background: 'none', border: 'none', color: '#EF4444', 
                fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '4px',
                marginTop: '8px', cursor: 'pointer'
              }}
            >
              <Trash2 size={12} /> Clear Document
            </button>
          </div>
        </div>
      )}

      <div className="privacy-card">
        <ShieldCheck size={20} color="var(--color-accent-gold)" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <p style={{ color: 'var(--color-text-primary)', fontWeight: 500, marginBottom: '4px' }}>
            Your conversations are private and secure.
          </p>
          <p style={{ fontSize: '0.75rem' }}>Built for confidentiality.</p>
        </div>
      </div>
    </aside>
  );
};
