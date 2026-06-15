import { useState, useEffect, useCallback } from 'react';
import './App.css';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { InputArea } from './components/InputArea';
import { api } from './services/api';

export type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
};

export type CaseStatus = {
  hasCase: boolean;
  summary: string | null;
  fileName?: string;
};

export type Conversation = {
  id: string;
  title: string;
  timestamp: number;
  messages: Message[];
  caseStatus: CaseStatus;
};

// Multilingual auto-detection logic (ported from Streamlit app)
const SUPPORTED_REPLY_LANGUAGES: Record<string, string> = {
  "Auto (Match My Input)": "auto",
  "English": "en",
  "Hindi": "hindi",
  "Telugu": "telugu",
  "Tamil": "tamil",
  "Kannada": "kannada",
  "Malayalam": "malayalam",
  "Marathi": "marathi",
  "Bengali": "bengali",
  "Gujarati": "gujarati",
  "Urdu": "urdu",
  "Arabic": "arabic",
  "Punjabi": "punjabi",
};

const detectInputLanguage = (query: string): string => {
  const explicitMatch = query.match(/\bin\s+([A-Za-z]+)\b/i);
  if (explicitMatch) {
    const explicitLang = explicitMatch[1].toLowerCase();
    for (const [label, code] of Object.entries(SUPPORTED_REPLY_LANGUAGES)) {
      if (label.toLowerCase() === explicitLang || code === explicitLang) {
        return code;
      }
    }
  }

  const scriptPatterns: [RegExp, string][] = [
    [/[\u0900-\u097F]/, "hindi"],
    [/[\u0C00-\u0C7F]/, "telugu"],
    [/[\u0B80-\u0BFF]/, "tamil"],
    [/[\u0C80-\u0CFF]/, "kannada"],
    [/[\u0D00-\u0D7F]/, "malayalam"],
    [/[\u0980-\u09FF]/, "bengali"],
    [/[\u0A00-\u0A7F]/, "punjabi"],
    [/[\u0600-\u06FF]/, "urdu"],
    [/[\u0750-\u077F]/, "arabic"],
    [/[\u0A80-\u0AFF]/, "gujarati"],
  ];

  for (const [pattern, langCode] of scriptPatterns) {
    if (pattern.test(query)) {
      return langCode;
    }
  }
  return "en";
};

function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  
  const [isLoading, setIsLoading] = useState(false);
  const [language, setLanguage] = useState('Auto (Match My Input)');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  // Active conversation states
  const activeConversation = conversations.find(c => c.id === currentConversationId);
  const messages = activeConversation?.messages || [];
  const caseStatus = activeConversation?.caseStatus || { hasCase: false, summary: null };

  // Load initial case status on mount
  useEffect(() => {
    api.getCaseStatus().then((status) => {
      if (status.has_case && conversations.length === 0) {
        // If there's an active case on the backend but no conversation, initialize one
        handleNewChat({ hasCase: true, summary: status.summary });
      }
    }).catch(console.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleNewChat = useCallback((initialCaseStatus: CaseStatus = { hasCase: false, summary: null }) => {
    const newId = Date.now().toString();
    const newConv: Conversation = {
      id: newId,
      title: 'New Conversation',
      timestamp: Date.now(),
      messages: [],
      caseStatus: initialCaseStatus
    };
    setConversations(prev => [newConv, ...prev]);
    setCurrentConversationId(newId);
  }, []);

  // Ensure there's always an active conversation if messages are empty and user tries to type
  const ensureActiveConversation = useCallback(() => {
    if (!currentConversationId) {
      const newId = Date.now().toString();
      const newConv: Conversation = {
        id: newId,
        title: 'New Conversation',
        timestamp: Date.now(),
        messages: [],
        caseStatus: { hasCase: false, summary: null }
      };
      setConversations(prev => [newConv, ...prev]);
      setCurrentConversationId(newId);
      return newId;
    }
    return currentConversationId;
  }, [currentConversationId]);

  const handleSendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return;

    const targetConvId = ensureActiveConversation();
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: text };
    
    setConversations(prev => prev.map(conv => {
      if (conv.id === targetConvId) {
        const newMessages = [...conv.messages, userMsg];
        // Auto title generation on first message
        let title = conv.title;
        if (conv.messages.length === 0) {
          title = text.split(' ').slice(0, 4).join(' ') + (text.split(' ').length > 4 ? '...' : '');
        }
        return { ...conv, title, messages: newMessages, timestamp: Date.now() };
      }
      return conv;
    }));
    
    setIsLoading(true);

    try {
      const history: [string, string][] = [];
      let lastUserMsg = '';
      for (const msg of messages) {
        if (msg.role === 'user') {
          lastUserMsg = msg.content;
        } else if (msg.role === 'assistant' && lastUserMsg) {
          history.push([lastUserMsg, msg.content]);
          lastUserMsg = '';
        }
      }

      // Multilingual logic
      let apiText = text;
      let targetLang = SUPPORTED_REPLY_LANGUAGES[language];
      if (targetLang === 'auto') {
        targetLang = detectInputLanguage(text);
      }
      
      // If language is not english, append the "in {language}" so the backend translates it
      if (targetLang !== 'en' && targetLang !== 'auto') {
        // Just checking if the user already appended it
        if (!/\bin\s+[a-z]+\b/i.test(text)) {
          apiText = `${text} (in ${targetLang})`;
        }
      }

      const response = await api.chat(apiText, history);
      
      const assistantMsg: Message = { 
        id: (Date.now() + 1).toString(), 
        role: 'assistant', 
        content: response.answer 
      };
      
      setConversations(prev => prev.map(conv => 
        conv.id === targetConvId 
          ? { ...conv, messages: [...conv.messages, assistantMsg] }
          : conv
      ));
    } catch (error) {
      console.error("Failed to send message", error);
      const errorMsg: Message = { 
        id: (Date.now() + 1).toString(), 
        role: 'assistant', 
        content: "I'm sorry, I encountered an error. Please try again." 
      };
      setConversations(prev => prev.map(conv => 
        conv.id === targetConvId 
          ? { ...conv, messages: [...conv.messages, errorMsg] }
          : conv
      ));
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, messages, language, ensureActiveConversation]);

  const handleUploadDocument = useCallback(async (file: File) => {
    setIsUploading(true);
    setUploadProgress(10);
    try {
      const response = await api.uploadCase(file, (progress) => {
        setUploadProgress(progress);
      });
      
      const newStatus = {
        hasCase: true,
        summary: response.summary,
        fileName: file.name
      };

      const targetConvId = ensureActiveConversation();
      setConversations(prev => prev.map(conv => 
        conv.id === targetConvId 
          ? { ...conv, caseStatus: newStatus }
          : conv
      ));

    } catch (error) {
      console.error("Upload failed", error);
      alert("Failed to analyze the document. Please try again.");
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  }, [ensureActiveConversation]);

  const handleClearDocument = useCallback(async () => {
    try {
      await api.clearCase();
      if (currentConversationId) {
        setConversations(prev => prev.map(conv => 
          conv.id === currentConversationId 
            ? { ...conv, caseStatus: { hasCase: false, summary: null, fileName: undefined } }
            : conv
        ));
      }
    } catch (error) {
      console.error("Failed to clear document", error);
    }
  }, [currentConversationId]);

  return (
    <div className="app-container">
      <div className="watermark-bg" />
      <Sidebar 
        language={language} 
        setLanguage={setLanguage} 
        onClearDocument={handleClearDocument}
        caseStatus={caseStatus}
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={setCurrentConversationId}
        onNewChat={() => handleNewChat()}
      />
      <main className="main-content">
        <ChatArea 
          messages={messages} 
          isLoading={isLoading} 
          caseStatus={caseStatus}
        />
        <InputArea 
          onSendMessage={handleSendMessage} 
          onUploadDocument={handleUploadDocument}
          isLoading={isLoading}
          isUploading={isUploading}
          uploadProgress={uploadProgress}
        />
      </main>
    </div>
  );
}

export default App;
