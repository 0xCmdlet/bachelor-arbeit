import { useState, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, Bot, User, Loader2, MessageSquare } from "lucide-react";
import { ragAPI } from "@/lib/api";
import { ConversationList } from "./ConversationList";

interface Message {
  id: string;
  content: string;
  type: "user" | "assistant";
  timestamp: Date;
  sources?: Array<{
    filename: string;
    score: number;
    text: string;
    chunk_id: string;
    chunk_index?: number;
    mime?: string;
    headers?: Record<string, string>;
  }>;
  isLoading?: boolean;
}

// Helper function to extract clean preview text from source content
const getCleanPreviewText = (text: string, maxLength: number = 150): string => {
  if (!text) return '';

  // Split into lines
  const lines = text.split('\n');

  // Filter out markdown tables and metadata
  const contentLines = lines.filter(line => {
    const trimmed = line.trim();

    // Skip empty lines
    if (!trimmed) return false;

    // Skip markdown table lines (contain multiple | characters)
    if (trimmed.includes('|') && trimmed.split('|').length > 2) return false;

    // Skip table separator lines (dashes and pipes)
    if (/^[\s\-|]+$/.test(trimmed)) return false;

    // Skip common metadata patterns
    if (/^(Date|Author|Speculator|Contributors|Created|Modified|Title)[\s:]/i.test(trimmed)) {
      return false;
    }

    return true;
  });

  // Join remaining lines and take first meaningful content
  const cleanText = contentLines.join(' ').trim();

  // If we have clean text, use it; otherwise fall back to original
  const preview = cleanText || text.replace(/\|/g, '').trim();

  // Truncate to max length
  if (preview.length <= maxLength) return preview;

  return preview.substring(0, maxLength) + '...';
};

export function ChatInterface() {
  const queryClient = useQueryClient();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content: "Hello! I'm your RAG assistant. How can I help you today?",
      type: "assistant",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Load thread ID from localStorage on mount
  useEffect(() => {
    const savedThreadId = localStorage.getItem("conversationThreadId");
    if (savedThreadId) {
      setThreadId(savedThreadId);
      console.log("📂 Loaded thread ID from storage:", savedThreadId);
      // Load history for saved thread
      loadConversationHistory(savedThreadId);
    }
  }, []);

  // Save thread ID to localStorage when it changes
  useEffect(() => {
    if (threadId) {
      localStorage.setItem("conversationThreadId", threadId);
      console.log("💾 Saved thread ID to storage:", threadId);
    }
  }, [threadId]);

  // Helper to generate title from first message
  const generateTitle = (message: string): string => {
    const maxLength = 50;
    if (message.length <= maxLength) return message;

    // Truncate at word boundary
    const truncated = message.substring(0, maxLength);
    const lastSpace = truncated.lastIndexOf(" ");
    return (
      (lastSpace > 0 ? truncated.substring(0, lastSpace) : truncated) + "..."
    );
  };

  // Load conversation history
  const loadConversationHistory = async (threadIdToLoad: string) => {
    setIsLoadingHistory(true);
    try {
      console.log("📜 Loading conversation history for:", threadIdToLoad);
      const historyData = await ragAPI.getHistory(threadIdToLoad);

      // Convert history to Message format, filtering out system/tool messages
      const loadedMessages: Message[] = historyData.messages
        .filter((msg) => {
          // Only include user and assistant messages with content
          return (
            (msg.role === "user" || msg.role === "assistant") &&
            msg.content &&
            msg.content.trim().length > 0
          );
        })
        .map((msg, idx) => ({
          id: `${threadIdToLoad}-${idx}`,
          content: msg.content,
          type: msg.role === "user" ? "user" : "assistant",
          timestamp: new Date(),
        }));

      if (loadedMessages.length > 0) {
        setMessages(loadedMessages);
        console.log("✅ Loaded", loadedMessages.length, "messages");
      } else {
        // No history yet, show welcome message
        setMessages([
          {
            id: "1",
            content: "Hello! I'm your RAG assistant. How can I help you today?",
            type: "assistant",
            timestamp: new Date(),
          },
        ]);
      }
    } catch (error) {
      console.error("❌ Failed to load history:", error);
      // On error, show welcome message
      setMessages([
        {
          id: "1",
          content: "Hello! I'm your RAG assistant. How can I help you today?",
          type: "assistant",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const chatMutation = useMutation({
    mutationFn: ragAPI.chat,
    onSuccess: (data) => {
      console.log("📦 Full chat response:", data);
      console.log("📝 Response length:", data.response?.length);
      console.log("🧵 Thread ID:", data.thread_id);
      console.log("🔧 Tools used:", data.tools_used);
      console.log("💭 Message count:", data.message_count);

      // Save thread ID for future messages
      setThreadId(data.thread_id);

      // Remove loading message
      setMessages((prev) => prev.filter((msg) => !msg.isLoading));

      // Add assistant response
      const assistantMessage: Message = {
        id: Date.now().toString(),
        content: data.response,
        type: "assistant",
        timestamp: new Date(),
        sources: data.sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // Refetch conversations list to update message counts
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
    onError: (error) => {
      // Remove loading message
      setMessages((prev) => prev.filter((msg) => !msg.isLoading));

      // Add error message
      const errorMessage: Message = {
        id: Date.now().toString(),
        content:
          "Sorry, I encountered an error processing your request. Please try again.",
        type: "assistant",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
      console.error("Chat error:", error);
    },
  });

  const handleSendMessage = async () => {
    if (!input.trim() || chatMutation.isPending) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      type: "user",
      timestamp: new Date(),
    };

    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      content: "Thinking...",
      type: "assistant",
      timestamp: new Date(),
      isLoading: true,
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);

    // If this is first message in new conversation, create conversation with auto-generated title
    const isFirstMessage = !threadId;
    if (isFirstMessage) {
      const title = generateTitle(input);
      try {
        const newConversation = await ragAPI.createConversation(title);
        console.log(
          "📝 Created new conversation:",
          newConversation.thread_id,
          "with title:",
          title
        );

        // Use the new thread_id
        chatMutation.mutate({
          message: input,
          thread_id: newConversation.thread_id,
        });
      } catch (error) {
        console.error(
          "❌ Failed to create conversation, proceeding without title:",
          error
        );
        // Fallback: send message without pre-creating conversation
        chatMutation.mutate({
          message: input,
          thread_id: undefined,
        });
      }
    } else {
      // Use conversational chat endpoint with existing thread_id
      chatMutation.mutate({
        message: input,
        thread_id: threadId,
      });
    }

    setInput("");
  };

  const handleSelectConversation = (selectedThreadId: string) => {
    if (selectedThreadId === threadId) return; // Already selected

    setThreadId(selectedThreadId);
    loadConversationHistory(selectedThreadId);
    setIsSidebarOpen(false); // Close sidebar on mobile after selection
  };

  const handleNewConversation = () => {
    // Clear thread ID and messages
    setThreadId(null);
    localStorage.removeItem("conversationThreadId");
    setMessages([
      {
        id: "1",
        content: "Hello! I'm your RAG assistant. How can I help you today?",
        type: "assistant",
        timestamp: new Date(),
      },
    ]);
    setIsSidebarOpen(false); // Close sidebar on mobile
    console.log("🆕 Started new conversation");
  };

  const handleDeleteConversation = (deletedThreadId: string) => {
    // If deleted conversation is current, start new conversation
    if (deletedThreadId === threadId) {
      handleNewConversation();
    }
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Conversation List Sidebar */}
      <ConversationList
        currentThreadId={threadId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
      />

      {/* Main Chat Area */}
      <div className="flex flex-col flex-1 max-w-6xl mx-auto w-full">
        {/* Header - Floating Glass Design */}
        <div className="sticky top-4 z-10 mx-4 mt-4 mb-2">
          <div className="glass-strong rounded-2xl shadow-2xl border border-primary/20">
            <div className="flex justify-between items-center px-8 py-5">
              <div className="flex items-center gap-4 ml-12 lg:ml-0">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-primary to-purple-600 rounded-2xl blur-xl opacity-60 animate-pulse-glow"></div>
                  <div className="relative flex items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-br from-primary to-purple-600 shadow-lg">
                    <Bot className="h-6 w-6 text-white" />
                  </div>
                </div>
                <div>
                  <h1 className="text-2xl font-bold gradient-text tracking-tight">
                    RAG Assistant
                  </h1>
                  <p className="text-sm text-muted-foreground font-medium">
                    AI-powered intelligent search
                  </p>
                </div>
              </div>
              {threadId && (
                <div className="hidden sm:flex items-center gap-2.5 px-4 py-2 glass rounded-xl border border-primary/30">
                  <MessageSquare className="h-4 w-4 text-primary" />
                  <span className="text-xs font-mono font-semibold text-primary">
                    {threadId.substring(0, 8)}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 px-6 py-8 overflow-y-auto custom-scrollbar max-h-[calc(100vh-280px)]">
          {isLoadingHistory ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center animate-slide-up-bounce">
                <div className="relative inline-block">
                  <div className="absolute inset-0 bg-primary/30 rounded-full blur-2xl animate-pulse"></div>
                  <Loader2 className="relative h-12 w-12 animate-spin text-primary" />
                </div>
                <p className="mt-4 text-base font-semibold text-foreground">
                  Loading conversation...
                </p>
                <p className="text-sm text-muted-foreground">Just a moment</p>
              </div>
            </div>
          ) : (
            <div className="space-y-8 max-w-4xl mx-auto pb-4">
              {messages.map((message, index) => (
                <div
                  key={message.id}
                  className={`flex gap-4 animate-slide-up-bounce ${
                    message.type === "user" ? "justify-end" : "justify-start"
                  }`}
                  style={{ animationDelay: `${index * 80}ms` }}
                >
                  {message.type === "assistant" && (
                    <div className="flex items-start pt-1">
                      <div className="relative group">
                        <div className="absolute inset-0 bg-gradient-to-br from-primary/30 to-purple-600/30 rounded-2xl blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                        <div className="relative flex items-center justify-center w-10 h-10 rounded-2xl bg-gradient-to-br from-primary/20 to-purple-600/20 border border-primary/30 shadow-lg">
                          <Bot className="h-5 w-5 text-primary" />
                        </div>
                      </div>
                    </div>
                  )}

                  <div
                    className={`flex flex-col gap-2 ${
                      message.type === "user" ? "items-end" : "items-start"
                    } max-w-[80%] sm:max-w-[70%]`}
                  >
                    <div
                      className={`relative group ${
                        message.type === "user" ? "ml-auto" : ""
                      }`}
                    >
                      {/* Glow effect */}
                      <div
                        className={`absolute inset-0 rounded-3xl blur-xl opacity-0 group-hover:opacity-60 transition-all duration-500 ${
                          message.type === "user"
                            ? "bg-gradient-to-br from-primary/40 to-purple-600/40"
                            : "bg-primary/20"
                        }`}
                      ></div>

                      {/* Message bubble */}
                      <div
                        className={`relative ${
                          message.type === "user"
                            ? "glass-strong bg-gradient-to-br from-primary to-purple-600 text-white shadow-2xl"
                            : "glass border border-primary/10 shadow-xl"
                        } rounded-3xl p-5 transition-all duration-300 hover:scale-[1.02]`}
                      >
                        <div className="flex items-start gap-3">
                          <p
                            className={`text-base leading-relaxed whitespace-pre-wrap font-medium ${
                              message.type === "user"
                                ? "text-white"
                                : "text-foreground"
                            }`}
                          >
                            {message.content}
                          </p>
                          {message.isLoading && (
                            <Loader2 className="h-5 w-5 animate-spin shrink-0 mt-0.5 text-primary" />
                          )}
                        </div>

                        {message.sources && message.sources.length > 0 && (
                          <div className="mt-4 pt-4 border-t border-white/10">
                            <p className="text-xs font-bold text-white/90 mb-3 flex items-center gap-2 uppercase tracking-wider">
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent animate-pulse"></span>
                              Sources ({message.sources.length})
                            </p>
                            <div className="space-y-3">
                              {message.sources
                                .slice(0, 3)
                                .map((source, idx) => (
                                  <div
                                    key={idx}
                                    className="bg-white/10 backdrop-blur-sm p-3 rounded-2xl border border-white/20 hover:bg-white/20 transition-all duration-200 hover:scale-[1.02]"
                                  >
                                    {/* Header with filename and score */}
                                    <div className="flex items-center justify-between mb-2 pb-2 border-b border-white/10">
                                      <div className="flex items-center gap-2 flex-1 min-w-0">
                                        <span className="text-accent text-sm">📄</span>
                                        <span className="text-xs font-bold text-white truncate">
                                          {source.filename || 'Unknown Document'}
                                        </span>
                                      </div>
                                      <span className="text-xs font-bold text-accent ml-2 shrink-0">
                                        {Math.round(source.score * 100)}% match
                                      </span>
                                    </div>
                                    {/* Text preview */}
                                    <p className="text-xs text-white/80 leading-relaxed">
                                      {getCleanPreviewText(source.text, 150)}
                                    </p>
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    <span className="text-xs text-muted-foreground px-2 font-medium">
                      {message.timestamp.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>

                  {message.type === "user" && (
                    <div className="flex items-start pt-1">
                      <div className="relative group">
                        <div className="absolute inset-0 bg-gradient-to-br from-primary/40 to-purple-600/40 rounded-2xl blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                        <div className="relative flex items-center justify-center w-10 h-10 rounded-2xl bg-gradient-to-br from-primary to-purple-600 shadow-xl border border-white/20">
                          <User className="h-5 w-5 text-white" />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Input Area - Floating Glass Design */}
        <div className="p-4 pb-6">
          <div className="max-w-4xl mx-auto">
            <div className="relative">
              {/* Ambient glow */}
              <div className="absolute -inset-2 bg-gradient-to-r from-primary/20 via-purple-600/20 to-accent/20 rounded-3xl blur-2xl opacity-50 animate-pulse"></div>

              {/* Input container */}
              <div className="relative glass-strong rounded-3xl p-2 shadow-2xl border border-primary/20">
                <div className="flex items-end gap-3">
                  <div className="flex-1 relative">
                    <Input
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSendMessage();
                        }
                      }}
                      placeholder="Ask me anything about your documents..."
                      className="bg-transparent border-0 min-h-[56px] text-base px-6 py-4 focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-muted-foreground/60 font-medium"
                      disabled={chatMutation.isPending}
                    />
                    {input.length > 0 && (
                      <div className="absolute right-4 bottom-4">
                        <span className="glass px-3 py-1 rounded-full text-xs font-mono font-bold text-primary border border-primary/30">
                          {input.length}
                        </span>
                      </div>
                    )}
                  </div>
                  <Button
                    onClick={handleSendMessage}
                    disabled={!input.trim() || chatMutation.isPending}
                    size="icon"
                    className="h-14 w-14 rounded-2xl shrink-0 bg-gradient-to-br from-primary to-purple-600 hover:from-primary/90 hover:to-purple-600/90 shadow-xl transition-all duration-300 hover:scale-110 active:scale-95 disabled:opacity-50 disabled:hover:scale-100 border-0 relative group"
                  >
                    {/* Button glow */}
                    <div className="absolute inset-0 bg-gradient-to-br from-primary to-purple-600 rounded-2xl blur-lg opacity-50 group-hover:opacity-75 transition-opacity"></div>

                    {chatMutation.isPending ? (
                      <Loader2 className="relative h-6 w-6 animate-spin text-white" />
                    ) : (
                      <Send className="relative h-6 w-6 text-white" />
                    )}
                  </Button>
                </div>
              </div>
            </div>
            <p className="text-xs text-muted-foreground/80 mt-3 text-center font-medium tracking-wide">
              Press{" "}
              <kbd className="px-2 py-0.5 bg-muted/50 rounded border border-primary/20 font-mono text-xs">
                Enter
              </kbd>{" "}
              to send •{" "}
              <kbd className="px-2 py-0.5 bg-muted/50 rounded border border-primary/20 font-mono text-xs">
                Shift + Enter
              </kbd>{" "}
              for new line
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
