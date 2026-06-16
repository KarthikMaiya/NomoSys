import { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import { Sidebar } from './components/Sidebar';
import { ChatArea } from './components/ChatArea';
import { InputArea } from './components/InputArea';
import { api } from './services/api';

export type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  /** Track which request produced this response */
  requestId?: string;
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
  language: string; // Per-conversation language preference
};

export type LoadingStatus = {
  active: boolean;
  phase: string;
};

// ─── Language Mapping ─────────────────────────────────────────────────
// Keys = what the backend's detect_output_language() recognises after "in <word>"
// Values = what deep_translator / GoogleTranslator accepts as target_lang
// The backend regex is: re.search(r"in (\w+)", query, re.IGNORECASE)
// So we must append " in Hindi" (not "(in hindi)") to the query text.
const LANGUAGE_TO_BACKEND_WORD: Record<string, string> = {
  "English": "English",
  "Hindi": "Hindi",
  "Telugu": "Telugu",
  "Tamil": "Tamil",
  "Kannada": "Kannada",
  "Malayalam": "Malayalam",
  "Marathi": "Marathi",
  "Bengali": "Bengali",
  "Gujarati": "Gujarati",
  "Urdu": "Urdu",
  "Arabic": "Arabic",
  "Punjabi": "Punjabi",
};

// Script-based auto-detection (mirrors chatbot_backend.py patterns)
const detectInputLanguageWord = (query: string): string | null => {
  const scriptPatterns: [RegExp, string][] = [
    [/[\u0900-\u097F]/, "Hindi"],
    [/[\u0C00-\u0C7F]/, "Telugu"],
    [/[\u0B80-\u0BFF]/, "Tamil"],
    [/[\u0C80-\u0CFF]/, "Kannada"],
    [/[\u0D00-\u0D7F]/, "Malayalam"],
    [/[\u0980-\u09FF]/, "Bengali"],
    [/[\u0A00-\u0A7F]/, "Punjabi"],
    [/[\u0600-\u06FF]/, "Urdu"],
    [/[\u0750-\u077F]/, "Arabic"],
    [/[\u0A80-\u0AFF]/, "Gujarati"],
  ];
  for (const [pattern, langWord] of scriptPatterns) {
    if (pattern.test(query)) return langWord;
  }
  return null; // English / not detected
};

const LOADING_PHASES = [
  "Analyzing legal query...",
  "Searching legal knowledge base...",
  "Retrieving relevant documents...",
  "Generating legal response...",
];

function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<LoadingStatus>({ active: false, phase: '' });
  const [language, setLanguage] = useState('Auto (Match My Input)');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  // Abort controller for in-flight requests — prevents stale responses
  const abortControllerRef = useRef<AbortController | null>(null);
  // Track the active request to prevent cross-conversation contamination
  const activeRequestRef = useRef<{ conversationId: string; requestId: string } | null>(null);
  // Loading phase animation interval
  const loadingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Active conversation derived state
  const activeConversation = conversations.find(c => c.id === currentConversationId);
  const messages = activeConversation?.messages || [];
  const caseStatus = activeConversation?.caseStatus || { hasCase: false, summary: null };

  // Load initial case status on mount — only once
  useEffect(() => {
    api.getCaseStatus().then((status) => {
      if (status.has_case) {
        handleNewChat({ hasCase: true, summary: status.summary });
      }
    }).catch(() => { /* backend not running yet, that's fine */ });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cleanup abort controller and interval on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      if (loadingIntervalRef.current) clearInterval(loadingIntervalRef.current);
    };
  }, []);

  const startLoadingAnimation = useCallback(() => {
    let phaseIndex = 0;
    setLoadingStatus({ active: true, phase: LOADING_PHASES[0] });
    loadingIntervalRef.current = setInterval(() => {
      phaseIndex = (phaseIndex + 1) % LOADING_PHASES.length;
      setLoadingStatus({ active: true, phase: LOADING_PHASES[phaseIndex] });
    }, 2500);
  }, []);

  const stopLoadingAnimation = useCallback(() => {
    setLoadingStatus({ active: false, phase: '' });
    if (loadingIntervalRef.current) {
      clearInterval(loadingIntervalRef.current);
      loadingIntervalRef.current = null;
    }
  }, []);

  const handleNewChat = useCallback((initialCaseStatus: CaseStatus = { hasCase: false, summary: null }) => {
    const newId = Date.now().toString();
    const newConv: Conversation = {
      id: newId,
      title: 'New Conversation',
      timestamp: Date.now(),
      messages: [],
      caseStatus: initialCaseStatus,
      language: 'Auto (Match My Input)',
    };
    setConversations(prev => [newConv, ...prev]);
    setCurrentConversationId(newId);
  }, []);

  const ensureActiveConversation = useCallback((): string => {
    if (!currentConversationId) {
      const newId = Date.now().toString();
      const newConv: Conversation = {
        id: newId,
        title: 'New Conversation',
        timestamp: Date.now(),
        messages: [],
        caseStatus: { hasCase: false, summary: null },
        language: 'Auto (Match My Input)',
      };
      setConversations(prev => [newConv, ...prev]);
      setCurrentConversationId(newId);
      return newId;
    }
    return currentConversationId;
  }, [currentConversationId]);

  const handleSendMessage = useCallback(async (text: string) => {
    if (!text.trim() || loadingStatus.active) return;

    // Cancel any in-flight request to prevent stale responses
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const targetConvId = ensureActiveConversation();
    const requestId = `req_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    activeRequestRef.current = { conversationId: targetConvId, requestId };

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: text, requestId };

    // Update conversation: add user message + auto-title on first message
    setConversations(prev => prev.map(conv => {
      if (conv.id === targetConvId) {
        const newMessages = [...conv.messages, userMsg];
        let title = conv.title;
        if (conv.messages.length === 0) {
          const words = text.split(/\s+/);
          title = words.slice(0, 5).join(' ') + (words.length > 5 ? '...' : '');
        }
        return { ...conv, title, messages: newMessages, timestamp: Date.now() };
      }
      return conv;
    }));

    // Start loading animation immediately
    startLoadingAnimation();

    try {
      // Build history from the CURRENT conversation's messages (use functional update pattern)
      let currentMessages: Message[] = [];
      setConversations(prev => {
        const conv = prev.find(c => c.id === targetConvId);
        if (conv) currentMessages = conv.messages;
        return prev; // no mutation
      });

      const history: [string, string][] = [];
      let lastUserContent = '';
      for (const msg of currentMessages) {
        if (msg.role === 'user') {
          lastUserContent = msg.content;
        } else if (msg.role === 'assistant' && lastUserContent) {
          history.push([lastUserContent, msg.content]);
          lastUserContent = '';
        }
      }

      // ─── Multilingual: build the correct API text ─────────────────────
      // The backend's detect_output_language() uses:
      //   re.search(r"in (\w+)", query, re.IGNORECASE)
      // So we must append " in Hindi" (without parentheses) to the question.
      let apiText = text;
      const selectedLang = language;

      if (selectedLang !== 'Auto (Match My Input)' && selectedLang !== 'English') {
        const backendWord = LANGUAGE_TO_BACKEND_WORD[selectedLang];
        if (backendWord) {
          // Only append if the user hasn't already typed "in <language>"
          const alreadyHasLang = new RegExp(`\\bin\\s+${backendWord}\\b`, 'i').test(text);
          if (!alreadyHasLang) {
            apiText = `${text} in ${backendWord}`;
          }
        }
      } else if (selectedLang === 'Auto (Match My Input)') {
        // Auto-detect from script
        const detectedWord = detectInputLanguageWord(text);
        if (detectedWord && detectedWord !== 'English') {
          const alreadyHasLang = new RegExp(`\\bin\\s+${detectedWord}\\b`, 'i').test(text);
          if (!alreadyHasLang) {
            apiText = `${text} in ${detectedWord}`;
          }
        }
      }

      const response = await api.chat(apiText, history, controller.signal);

      // ─── GUARD: Only render if this response belongs to the active request ───
      if (
        activeRequestRef.current?.requestId !== requestId ||
        activeRequestRef.current?.conversationId !== targetConvId
      ) {
        // Stale response — discard
        return;
      }

      if (controller.signal.aborted) return;

      const assistantMsg: Message = {
        id: `${Date.now()}_resp`,
        role: 'assistant',
        content: response.answer,
        requestId,
      };

      setConversations(prev => prev.map(conv =>
        conv.id === targetConvId
          ? { ...conv, messages: [...conv.messages, assistantMsg] }
          : conv
      ));
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === 'AbortError') return; // Intentional cancel

      // Only show error if this is still the active request
      if (activeRequestRef.current?.requestId !== requestId) return;

      console.error("Failed to send message", error);
      const errorMsg: Message = {
        id: `${Date.now()}_err`,
        role: 'assistant',
        content: "I'm sorry, I encountered an error connecting to the NomoSys servers. Please ensure the backend is running and try again.",
        requestId,
      };
      setConversations(prev => prev.map(conv =>
        conv.id === targetConvId
          ? { ...conv, messages: [...conv.messages, errorMsg] }
          : conv
      ));
    } finally {
      stopLoadingAnimation();
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
      activeRequestRef.current = null;
    }
  }, [loadingStatus.active, language, ensureActiveConversation, startLoadingAnimation, stopLoadingAnimation]);

  const handleUploadDocument = useCallback(async (file: File) => {
    setIsUploading(true);
    setUploadProgress(10);
    try {
      const response = await api.uploadCase(file, (progress) => {
        setUploadProgress(progress);
      });

      const newStatus: CaseStatus = {
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

  // When switching conversations, cancel in-flight requests
  const handleSelectConversation = useCallback((id: string) => {
    abortControllerRef.current?.abort();
    stopLoadingAnimation();
    activeRequestRef.current = null;
    setCurrentConversationId(id);
  }, [stopLoadingAnimation]);

  return (
    <div className="app-container">
      <div className="watermark-bg" aria-hidden="true" />
      <Sidebar
        language={language}
        setLanguage={setLanguage}
        onClearDocument={handleClearDocument}
        caseStatus={caseStatus}
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={() => handleNewChat()}
      />
      <main className="main-content">
        <ChatArea
          messages={messages}
          loadingStatus={loadingStatus}
          caseStatus={caseStatus}
        />
        <InputArea
          onSendMessage={handleSendMessage}
          onUploadDocument={handleUploadDocument}
          isLoading={loadingStatus.active}
          isUploading={isUploading}
          uploadProgress={uploadProgress}
        />
      </main>
    </div>
  );
}

export default App;
