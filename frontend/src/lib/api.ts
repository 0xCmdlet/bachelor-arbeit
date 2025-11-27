import axios from 'axios'

const API_BASE_URL = import.meta.env.PROD
  ? '/api' // In production, use reverse proxy
  : '/api' // In development, use Vite proxy

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes for RAG queries
})

// Legacy stateless query types
export interface QueryRequest {
  query: string
  top_k?: number
  min_score?: number
}

export interface QueryResponse {
  answer: string
  sources: Array<{
    text: string
    chunk_id: string
    score: number
  }>
  query: string
  processing_time: number
}

// Conversational chat types
export interface ChatRequest {
  message: string
  thread_id?: string
}

export interface ChatResponse {
  response: string
  thread_id: string
  tools_used: string[]
  message_count: number
  sources?: Array<{
    filename: string
    score: number
    text: string
    chunk_id: string
    chunk_index?: number
    mime?: string
    headers?: Record<string, string>
  }>
}

export interface ConversationHistoryResponse {
  thread_id: string
  messages: Array<{
    role: 'user' | 'assistant' | 'system'
    content: string
  }>
  count: number
}

export interface Conversation {
  thread_id: string
  title: string
  created_at: string | null
  updated_at: string | null
  metadata?: Record<string, unknown>
  message_count?: number
}

export interface ConversationsListResponse {
  conversations: Conversation[]
  count: number
}

export const ragAPI = {
  // Legacy stateless query endpoint (for comparison)
  query: async (request: QueryRequest): Promise<QueryResponse> => {
    console.log('🚀 Sending RAG query (stateless):', request)
    console.log('📍 API URL:', `${API_BASE_URL}/query`)

    try {
      const response = await apiClient.post<QueryResponse>('/query', request)
      console.log('✅ RAG response received:', response.data)
      console.log('📊 Processing time:', response.data.processing_time + 's')
      console.log('🔍 Sources found:', response.data.sources.length)
      return response.data
    } catch (error: unknown) {
      console.error('❌ RAG query failed:', error)
      if (error && typeof error === 'object' && 'response' in error) {
        const axiosError = error as { response: { status: number; data: unknown } }
        console.error('📋 Response status:', axiosError.response.status)
        console.error('📋 Response data:', axiosError.response.data)
      }
      throw error
    }
  },

  // NEW: Conversational chat endpoint with memory
  chat: async (request: ChatRequest): Promise<ChatResponse> => {
    console.log('💬 Sending chat message:', request)
    console.log('🧵 Thread ID:', request.thread_id || 'new conversation')

    try {
      const response = await apiClient.post<ChatResponse>('/chat', request)
      console.log('✅ Chat response received:', response.data)
      console.log('🧵 Thread ID:', response.data.thread_id)
      console.log('🔧 Tools used:', response.data.tools_used)
      console.log('💭 Messages in thread:', response.data.message_count)
      return response.data
    } catch (error: unknown) {
      console.error('❌ Chat failed:', error)
      if (error && typeof error === 'object' && 'response' in error) {
        const axiosError = error as { response: { status: number; data: unknown } }
        console.error('📋 Response status:', axiosError.response.status)
        console.error('📋 Response data:', axiosError.response.data)
      }
      throw error
    }
  },

  // Get conversation history
  getHistory: async (threadId: string): Promise<ConversationHistoryResponse> => {
    console.log('📜 Fetching conversation history for:', threadId)

    try {
      const response = await apiClient.get<ConversationHistoryResponse>(
        `/conversations/${threadId}/history`
      )
      console.log('✅ History retrieved:', response.data.count, 'messages')
      return response.data
    } catch (error: unknown) {
      console.error('❌ History fetch failed:', error)
      throw error
    }
  },

  // Delete conversation
  deleteConversation: async (threadId: string): Promise<void> => {
    console.log('🗑️ Deleting conversation:', threadId)

    try {
      await apiClient.delete(`/conversations/${threadId}`)
      console.log('✅ Conversation deleted')
    } catch (error: unknown) {
      console.error('❌ Delete failed:', error)
      throw error
    }
  },

  // List all conversations
  listConversations: async (limit: number = 50): Promise<ConversationsListResponse> => {
    console.log('📋 Fetching conversations list, limit:', limit)

    try {
      const response = await apiClient.get<ConversationsListResponse>(
        `/conversations?limit=${limit}`
      )
      console.log('✅ Conversations retrieved:', response.data.count, 'conversations')
      return response.data
    } catch (error: unknown) {
      console.error('❌ List conversations failed:', error)
      throw error
    }
  },

  // Create new conversation
  createConversation: async (title?: string): Promise<Conversation> => {
    console.log('➕ Creating new conversation with title:', title || 'untitled')

    try {
      const response = await apiClient.post<Conversation>('/conversations', {
        title: title || undefined
      })
      console.log('✅ Conversation created:', response.data.thread_id)
      return response.data
    } catch (error: unknown) {
      console.error('❌ Create conversation failed:', error)
      throw error
    }
  },

  health: async () => {
    console.log('💓 Checking health endpoint')
    try {
      const response = await apiClient.get('/health')
      console.log('✅ Health check response:', response.data)
      return response.data
    } catch (error) {
      console.error('❌ Health check failed:', error)
      throw error
    }
  }
}