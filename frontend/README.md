# RAG System Frontend

> **📦 Docker Container:** `frontend`
> **Port:** 80 (production), 3000 (development)
> **Purpose:** React-based web interface for RAG system
> **Related Services:** `rag-api`

## Overview

Modern React + TypeScript web interface for the RAG system, providing an intuitive chat-based experience with document sources and conversation management.

## Tech Stack

- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite
- **Styling:** TailwindCSS
- **State Management:** React Query (TanStack Query)
- **HTTP Client:** Axios
- **Icons:** Lucide React
- **Deployment:** Docker with nginx

## Features

- **Conversational Chat**: Multi-turn conversations with automatic context management
- **Thread Persistence**: Conversations saved across page refreshes (localStorage + PostgreSQL)
- **Document Sources**: View retrieved documents with relevance scores
- **New Chat**: Start fresh conversations with one click
- **Real-time Status**: System health and statistics monitoring
- **Responsive Design**: Works on desktop and mobile

## Quick Start

### Development Mode

```bash
cd frontend
npm install
npm run dev
```

Access at: http://localhost:3000

### Production Mode (Docker)

```bash
docker compose up -d frontend
```

Access at: http://localhost:80

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatInterface.tsx    # Main chat component
│   │   ├── ChatMessage.tsx      # Individual message display
│   │   ├── SourcesPanel.tsx     # Document sources sidebar
│   │   └── SystemStatus.tsx     # Health monitoring
│   ├── lib/
│   │   └── api.ts              # API client
│   ├── App.tsx                 # Root component
│   └── main.tsx                # Entry point
├── public/                     # Static assets
├── Dockerfile                  # Production build
├── nginx.conf                  # Production server config
├── vite.config.ts              # Vite configuration
├── tailwind.config.js          # TailwindCSS config
└── tsconfig.json               # TypeScript config
```

## Key Components

### ChatInterface

Main conversational interface with:
- Message history display
- Input field for user queries
- Thread management (new chat, continue conversation)
- Loading states and error handling

```typescript
const ChatInterface = () => {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);

  // Uses RAG API chat endpoint for conversational memory
  const chatMutation = useMutation({
    mutationFn: ragAPI.chat,
    onSuccess: (data) => {
      setThreadId(data.thread_id);
      // Update localStorage and message history
    }
  });

  return (
    // Chat UI with message list and input
  );
};
```

### API Client

Typed API client in `src/lib/api.ts`:

```typescript
export const ragAPI = {
  // Conversational endpoint (with memory)
  chat: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await axios.post("/chat", request);
    return response.data;
  },

  // Legacy stateless endpoint
  query: async (request: QueryRequest): Promise<QueryResponse> => {
    const response = await axios.post("/query", request);
    return response.data;
  },

  // System monitoring
  getHealth: async () => { /* ... */ },
  getStats: async () => { /* ... */ },

  // Conversation management
  getHistory: async (threadId: string) => { /* ... */ },
  deleteConversation: async (threadId: string) => { /* ... */ }
};
```

## API Integration

### Environment Configuration

The frontend connects to the RAG API via environment variables:

**Development (.env.development):**
```bash
VITE_API_URL=http://localhost:8080
```

**Production:**
API URL is configured via nginx proxy or environment variable at build time.

### Conversational Flow

1. **First Message:**
   - User types message
   - Frontend calls `/chat` without `thread_id`
   - Backend creates new conversation, returns `thread_id`
   - Frontend stores `thread_id` in state and localStorage

2. **Follow-up Messages:**
   - User types message
   - Frontend calls `/chat` with `thread_id`
   - Backend loads conversation history from PostgreSQL
   - Agent uses context to understand references ("it", "that", etc.)

3. **New Chat:**
   - User clicks "New Chat" button
   - Frontend clears `thread_id` and message history
   - Next message starts fresh conversation

### Message Format

**Chat Request:**
```typescript
interface ChatRequest {
  message: string;
  thread_id?: string;
}
```

**Chat Response:**
```typescript
interface ChatResponse {
  response: string;
  thread_id: string;
  tools_used?: string[];
  message_count?: number;
}
```

## Styling

Uses TailwindCSS utility classes for rapid development:

```tsx
<div className="flex flex-col h-screen bg-gray-50">
  <div className="flex-1 overflow-y-auto p-4">
    {/* Messages */}
  </div>
  <div className="border-t bg-white p-4">
    {/* Input */}
  </div>
</div>
```

Custom theme configuration in `tailwind.config.js`:
- Color palette matching brand
- Custom spacing and breakpoints
- Typography scale

## Docker Deployment

### Production Dockerfile

Multi-stage build for optimized image:

```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### nginx Configuration

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to backend
    location /api {
        proxy_pass http://rag-api:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## Development

### Hot Reload

Vite provides instant hot module replacement (HMR):

```bash
npm run dev
```

Changes to React components update instantly without page refresh.

### Type Checking

TypeScript strict mode enabled for type safety:

```bash
npm run type-check
```

### Linting

ESLint configuration for code quality:

```bash
npm run lint
```

### Building

Production build:

```bash
npm run build
```

Output in `dist/` folder, ready for nginx deployment.

## Testing Locally

### With Docker Backend

1. Start all services:
   ```bash
   docker compose up -d postgres minio qdrant ollama rag-api
   ```

2. Start frontend in dev mode:
   ```bash
   cd frontend
   npm run dev
   ```

3. Access at http://localhost:3000

### Full Docker Stack

```bash
docker compose up -d
```

Access at http://localhost:80

## Troubleshooting

### API Connection Issues

**Symptom:** Frontend can't reach API

**Solutions:**
```bash
# Check API is running
curl http://localhost:8080/health

# Check Docker network
docker network inspect final_default

# Verify VITE_API_URL in .env.development
echo $VITE_API_URL
```

### Build Errors

**Symptom:** `npm run build` fails

**Solutions:**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check TypeScript errors
npm run type-check
```

### CORS Issues

**Symptom:** Browser blocks API requests

**Solution:** Ensure rag-api has CORS enabled in `main.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Conversation Not Persisting

**Symptom:** Thread lost after page refresh

**Check:**
1. Browser localStorage: Open DevTools → Application → Local Storage
2. Look for `thread_id` key
3. Verify PostgreSQL has checkpoints table:
   ```bash
   docker exec -it postgres psql -U admin -d mydb -c "\dt"
   ```

## Documentation

- **[Project Structure](project-structure.md)**: Detailed architecture guide
- **[Conversational Update](CONVERSATIONAL_UPDATE.md)**: Chat feature implementation details
- **[API Documentation](../docs/conversational_api.md)**: Backend API reference

## Next Steps

1. **Implement streaming**: Real-time response streaming for better UX
2. **Add file upload**: Upload documents directly from frontend
3. **Conversation list**: Browse and resume previous conversations
4. **Export functionality**: Download conversation as markdown/PDF
5. **Dark mode**: Theme toggle for user preference
6. **Analytics dashboard**: Visualize usage metrics with Recharts

## Contributing

When adding new features:

1. Create TypeScript interfaces for API types
2. Use React Query for API state management
3. Follow TailwindCSS utility-first approach
4. Update API client in `src/lib/api.ts`
5. Add error handling and loading states
6. Test with Docker backend

## Related Documentation

- Backend API: [docs/conversational_api.md](../docs/conversational_api.md)
- System overview: [readme.md](../readme.md)
- Evaluation: [evaluation/README.md](../evaluation/README.md)
