import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Plus, Trash2, MessageSquare, Menu, X, Loader2 } from "lucide-react";
import { ragAPI } from "@/lib/api";
import { cn } from "@/lib/utils";
import { formatDistanceToNow } from "date-fns";

interface ConversationListProps {
  currentThreadId: string | null;
  onSelectConversation: (threadId: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (threadId: string) => void;
  isOpen: boolean;
  onToggle: () => void;
}

export function ConversationList({
  currentThreadId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  isOpen,
  onToggle,
}: ConversationListProps) {
  const queryClient = useQueryClient();

  // Fetch conversations
  const { data: conversationsData, isLoading } = useQuery({
    queryKey: ["conversations"],
    queryFn: () => ragAPI.listConversations(50),
    refetchInterval: 10000, // Refetch every 10 seconds to keep list updated
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: ragAPI.deleteConversation,
    onSuccess: (_, threadId) => {
      // Refetch conversations list
      queryClient.invalidateQueries({ queryKey: ["conversations"] });

      // If deleted current conversation, trigger new conversation
      if (threadId === currentThreadId) {
        onDeleteConversation(threadId);
      }
    },
  });

  const conversations = conversationsData?.conversations || [];

  const handleDelete = (e: React.MouseEvent, threadId: string) => {
    e.stopPropagation(); // Prevent conversation selection when clicking delete

    if (confirm("Are you sure you want to delete this conversation?")) {
      deleteMutation.mutate(threadId);
    }
  };

  const formatTimestamp = (timestamp: string | null) => {
    if (!timestamp) return "";
    try {
      return formatDistanceToNow(new Date(timestamp), { addSuffix: true });
    } catch {
      return "";
    }
  };

  return (
    <>
      {/* Mobile toggle button */}
      <Button
        onClick={onToggle}
        variant="ghost"
        size="icon"
        className="fixed top-4 left-4 z-50 lg:hidden glass-strong shadow-2xl rounded-2xl h-12 w-12 border border-primary/30 hover:bg-primary/10"
      >
        {isOpen ? <X className="h-5 w-5 text-primary" /> : <Menu className="h-5 w-5 text-primary" />}
      </Button>

      {/* Sidebar */}
      <div
        className={cn(
          "fixed inset-y-0 left-0 z-40 w-80 glass-strong border-r border-primary/10 shadow-2xl transform transition-all duration-500 ease-out lg:relative lg:translate-x-0",
          isOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="p-4 border-b border-primary/10">
            <div className="relative group">
              {/* Button glow */}
              <div className="absolute -inset-0.5 bg-gradient-to-r from-primary to-purple-600 rounded-2xl blur-lg opacity-30 group-hover:opacity-60 transition-opacity duration-300"></div>

              <Button
                onClick={onNewConversation}
                className="relative w-full bg-gradient-to-r from-primary to-purple-600 hover:from-primary/90 hover:to-purple-600/90 text-white border-0 shadow-xl transition-all duration-300 hover:scale-[1.03] active:scale-[0.97] h-12 rounded-2xl font-bold text-base"
                size="lg"
              >
                <Plus className="h-5 w-5 mr-2" />
                New Conversation
              </Button>
            </div>
          </div>

          {/* Conversations List */}
          <ScrollArea className="flex-1 custom-scrollbar">
            <div className="p-3 space-y-2">
              {isLoading && (
                <div className="text-center py-12 animate-slide-up-bounce">
                  <div className="relative inline-block">
                    <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl animate-pulse"></div>
                    <Loader2 className="relative h-10 w-10 animate-spin text-primary" />
                  </div>
                  <p className="mt-4 text-sm font-semibold text-foreground">Loading...</p>
                </div>
              )}

              {!isLoading && conversations.length === 0 && (
                <div className="text-center py-16 px-4 animate-slide-up-bounce">
                  <div className="relative inline-block mb-4">
                    <div className="absolute inset-0 bg-primary/10 rounded-3xl blur-2xl"></div>
                    <div className="relative glass rounded-3xl p-6 border border-primary/20">
                      <MessageSquare className="h-14 w-14 mx-auto text-primary" />
                    </div>
                  </div>
                  <p className="text-base font-bold text-foreground mb-2">
                    No conversations yet
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Click "New Conversation" to start
                  </p>
                </div>
              )}

              {conversations.map((conversation, index) => (
                <div
                  key={conversation.thread_id}
                  className="animate-slide-up-bounce"
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <div className="relative group">
                    {/* Hover glow */}
                    {currentThreadId === conversation.thread_id && (
                      <div className="absolute -inset-0.5 bg-gradient-to-r from-primary/40 to-purple-600/40 rounded-2xl blur-md opacity-60"></div>
                    )}

                    <div
                      className={cn(
                        "relative p-4 cursor-pointer glass border rounded-2xl transition-all duration-300 hover:scale-[1.02] active:scale-[0.98]",
                        currentThreadId === conversation.thread_id
                          ? "border-primary/50 bg-primary/10 shadow-lg"
                          : "border-primary/10 hover:border-primary/30 hover:bg-primary/5"
                      )}
                      onClick={() => onSelectConversation(conversation.thread_id)}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2.5 mb-2">
                            <div className={cn(
                              "p-1.5 rounded-lg transition-colors",
                              currentThreadId === conversation.thread_id
                                ? "bg-primary/20"
                                : "bg-primary/10"
                            )}>
                              <MessageSquare className={cn(
                                "h-4 w-4 transition-colors",
                                currentThreadId === conversation.thread_id
                                  ? "text-primary"
                                  : "text-primary/70"
                              )} />
                            </div>
                            <h3 className="text-sm font-bold truncate leading-tight text-foreground">
                              {conversation.title}
                            </h3>
                          </div>
                          <div className="flex items-center gap-2.5 text-xs font-medium">
                            <span className="flex items-center gap-1.5 text-muted-foreground">
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
                              {conversation.message_count || 0} msgs
                            </span>
                            {conversation.updated_at && (
                              <>
                                <span className="text-muted-foreground/50">•</span>
                                <span className="truncate text-muted-foreground">
                                  {formatTimestamp(conversation.updated_at)}
                                </span>
                              </>
                            )}
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-all flex-shrink-0 hover:bg-destructive/20 rounded-xl"
                          onClick={(e) => handleDelete(e, conversation.thread_id)}
                          disabled={deleteMutation.isPending}
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>

          {/* Footer */}
          <div className="p-4 border-t border-primary/10">
            <div className="glass rounded-2xl p-3 border border-primary/20">
              <p className="text-xs font-bold text-center gradient-text uppercase tracking-wider">
                {conversations.length} Conversation{conversations.length !== 1 ? "s" : ""}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-md z-30 lg:hidden transition-all duration-500 animate-in fade-in"
          onClick={onToggle}
        />
      )}
    </>
  );
}
