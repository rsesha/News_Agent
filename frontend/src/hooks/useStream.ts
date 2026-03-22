import { useState, useCallback, useRef } from 'react';

export interface Message {
  type: 'human' | 'ai';
  content: string;
  id?: string;
  time_taken?: number;
}

export interface StreamConfig {
  apiUrl: string;
  assistantId: string;
  messagesKey: string;
  onUpdateEvent?: (event: any) => void;
  onError?: (error: any) => void;
}

export function useStream<T>(config: StreamConfig) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const updateThread = useCallback((data: { messages: Message[] }) => {
    setMessages(data.messages);
  }, []);

  const stop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  }, []);

  const submit = useCallback(async (data: any) => {
    setIsLoading(true);
    console.log("DEBUG: Submitting search request", data);
    
    // Optimistically add user message if provided
    if (data.messages && data.messages.length > 0) {
      setMessages(prev => [...prev, ...data.messages]);
    }

    try {
      abortControllerRef.current = new AbortController();
      
      const fetchUrl = `${config.apiUrl.replace('localhost', '127.0.0.1')}/chat`;
      console.log(`DEBUG: Fetching from ${fetchUrl}`);

      const response = await fetch(fetchUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify(data),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim().startsWith('data: ')) {
            const jsonStr = line.trim().substring(6);
            console.log("DEBUG: Received event chunk", jsonStr);
            try {
              const event = JSON.parse(jsonStr);
              
              if (event.event === 'complete' && event.data.messages) {
                console.log("DEBUG: Research complete, adding AI messages", event.data.time_taken);
                setMessages(prev => {
                    const newMessages = [...prev];
                    event.data.messages.forEach((msg: Message) => {
                      if (!newMessages.find(m => m.content === msg.content && m.type === msg.type)) {
                        newMessages.push({ ...msg, time_taken: event.data.time_taken });
                      }
                    });
                    return newMessages;
                });
              }
              
              if (config.onUpdateEvent) {
                // Map event to what the frontend expects
                const mappedEvent: any = {};
                mappedEvent[event.event] = event.data;
                config.onUpdateEvent(mappedEvent);
              }
            } catch (e) {
              console.error('Error parsing SSE event:', e);
            }
          }
        }
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log('Stream aborted');
      } else {
        console.error('Stream error:', error);
        if (config.onError) config.onError(error);
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [config]);

  return {
    messages,
    isLoading,
    submit,
    stop,
    updateThread,
  };
}
