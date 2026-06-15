import React, { useState, useRef, memo, useCallback } from 'react';
import { Send, Plus } from 'lucide-react';

type InputAreaProps = {
  onSendMessage: (text: string) => void;
  onUploadDocument: (file: File) => void;
  isLoading: boolean;
  isUploading: boolean;
  uploadProgress: number;
};

export const InputArea: React.FC<InputAreaProps> = memo(({ 
  onSendMessage, 
  onUploadDocument, 
  isLoading,
  isUploading,
  uploadProgress
}) => {
  const [text, setText] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = useCallback((e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (text.trim() && !isLoading && !isUploading) {
      onSendMessage(text);
      setText('');
    }
  }, [text, isLoading, isUploading, onSendMessage]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onUploadDocument(e.target.files[0]);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  }, [onUploadDocument]);

  return (
    <div className="input-area-container">
      <form className="input-box" onSubmit={handleSubmit}>
        <input 
          type="file" 
          ref={fileInputRef} 
          style={{ display: 'none' }} 
          accept=".pdf,.txt"
          onChange={handleFileChange}
        />
        <button 
          type="button" 
          className="upload-button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading || isUploading}
          title="Upload Legal Document"
        >
          <Plus size={20} />
        </button>
        
        <input
          type="text"
          className="text-input"
          placeholder={isUploading ? `Uploading document (${uploadProgress}%)...` : "Ask a legal question..."}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading || isUploading}
        />
        
        <button 
          type="submit" 
          className="send-button"
          disabled={!text.trim() || isLoading || isUploading}
        >
          <Send size={18} style={{ marginLeft: '-2px' }} />
        </button>
      </form>
    </div>
  );
});
