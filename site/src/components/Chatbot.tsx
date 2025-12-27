import React, { useState, useEffect, useRef } from 'react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

const Chatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState<string | null>(null);
  const [queryMode, setQueryMode] = useState<'global' | 'selected_text'>('global');
  const [isTooltipVisible, setIsTooltipVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const tooltipTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Handle text selection
  useEffect(() => {
    const handleSelection = () => {
      if (tooltipTimeoutRef.current) {
        clearTimeout(tooltipTimeoutRef.current);
      }

      const selection = window.getSelection();
      if (selection && selection.toString().trim().length > 10) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();

        setTooltipPosition({
          top: rect.top + window.scrollY - 40,
          left: rect.left + window.scrollX + rect.width / 2 - 60
        });

        setIsTooltipVisible(true);
      } else {
        // Hide tooltip after a short delay to avoid flickering
        tooltipTimeoutRef.current = setTimeout(() => {
          setIsTooltipVisible(false);
        }, 300);
      }
    };

    document.addEventListener('selectionchange', handleSelection);
    document.addEventListener('mouseup', handleSelection);

    return () => {
      document.removeEventListener('selectionchange', handleSelection);
      document.removeEventListener('mouseup', handleSelection);
      if (tooltipTimeoutRef.current) {
        clearTimeout(tooltipTimeoutRef.current);
      }
    };
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = () => {
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Simulate API response (will be replaced with actual API call later)
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'This is a simulated response. The actual backend integration will be added later.'
      };
      setMessages(prev => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 1000);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleAskAboutSelectedText = () => {
    const selection = window.getSelection();
    const text = selection?.toString().trim();

    if (text && text.length > 10) {
      setSelectedText(text);
      setQueryMode('selected_text');
      setIsOpen(true);
      setIsTooltipVisible(false);
      selection?.empty(); // Clear selection
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const closeChat = () => {
    setIsOpen(false);
  };

  return (
    <>
      {/* Floating chat button */}
      <button
        className="chatbot-launcher"
        onClick={toggleChat}
        aria-label="Open chatbot"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
      </button>

      {/* Selected text tooltip */}
      {isTooltipVisible && (
        <button
          className="selected-text-tooltip"
          style={{
            position: 'absolute',
            top: `${tooltipPosition.top}px`,
            left: `${tooltipPosition.left}px`,
            zIndex: 10000
          }}
          onClick={handleAskAboutSelectedText}
          aria-label="Ask about selected text"
        >
          Ask about this
        </button>
      )}

      {/* Chat panel */}
      {isOpen && (
        <div className="chatbot-panel">
          <div className="chatbot-header">
            <div className="chatbot-title">Ask the Book</div>
            <button
              className="chatbot-close"
              onClick={closeChat}
              aria-label="Close chat"
            >
              ×
            </button>
          </div>

          {selectedText && queryMode === 'selected_text' && (
            <div className="selected-text-container">
              <div className="selected-text-content">
                <strong>Selected text:</strong> "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"
              </div>
              <button
                className="selected-text-clear"
                onClick={() => setSelectedText(null)}
                aria-label="Clear selected text"
              >
                ×
              </button>
            </div>
          )}

          <div className="chatbot-messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`chatbot-message ${message.role}`}
              >
                <div className="chatbot-message-content">{message.content}</div>
              </div>
            ))}
            {isLoading && (
              <div className="chatbot-message assistant">
                <div className="chatbot-message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chatbot-input-area">
            <textarea
              className="chatbot-input"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about the book..."
              disabled={isLoading}
              rows={1}
            />
            <button
              className="chatbot-send-button"
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              aria-label="Send message"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default Chatbot;