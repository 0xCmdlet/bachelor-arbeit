# RAG Frontend Implementation Guide

> **📦 Docker Container:** `frontend`
> **Port:** 3000 (development), 80 (production)
> **Purpose:** React-based web interface for RAG system
> **Related Services:** `rag-api`

This document outlines the complete implementation plan for the React frontend that will serve as the user interface for your RAG (Retrieval-Augmented Generation) system.

## 🎯 Project Overview

### Purpose

Create a modern, user-friendly web interface for your RAG system that transforms API calls into an intuitive chat-based experience with visual document sources and system monitoring.

### Architecture Integration

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │◄──►│    RAG API      │◄──►│  Ollama + Data  │
│   (Port 3000)   │    │   (Port 8080)   │    │   Services      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technical Stack

### Frontend Technologies

- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite (fast development and builds)
- **Styling:** TailwindCSS (utility-first, rapid development)
- **State Management:** React Query (API state + caching)
- **HTTP Client:** Axios
- **Icons:** Lucide React
- **Charts:** Recharts (for analytics dashboard)
- **Deployment:** Docker container with nginx

### Development Tools

- **Linting:** ESLint + Prettier
- **Type Checking:** TypeScript strict mode
- **Hot Reload:** Vite dev server
- **Package Manager:** npm

## 📁 Project Structure

```
frontend/
├── Dockerfile
├── Dockerfile.prod
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── vite.config.ts
├── index.html
├── public/
│   └── favicon.ico
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   ├── index.css
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── Layout.tsx
│   │   ├── chat/
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── QueryInput.tsx
│   │   │   └── SourcesList.tsx
│   │   ├── dashboard/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── HealthStatus.tsx
│   │   │   ├── SystemStats.tsx
│   │   │   └── ModelInfo.tsx
│   │   ├── settings/
│   │   │   ├── Settings.tsx
│   │   │   └── QueryControls.tsx
│   │   └── ui/
│   │       ├── Button.tsx
│   │       ├── Card.tsx
│   │       ├── Input.tsx
│   │       └── Loading.tsx
│   ├── api/
│   │   ├── ragApi.ts
│   │   └── types.ts
│   ├── hooks/
│   │   ├── useQuery.ts
│   │   ├── useHealth.ts
│   │   └── useStats.ts
│   ├── utils/
│   │   ├── formatters.ts
│   │   └── constants.ts
│   └── types/
│       └── index.ts
└── nginx.conf
```

## 🔌 API Integration

### Available Endpoints

Your RAG API provides the following endpoints that the frontend will consume:

#### 1. Query Endpoint

```typescript
POST /query
Request: {
  query: string;
  top_k?: number;
  min_score?: number;
}
Response: {
  answer: string;
  sources: Source[];
  query: string;
  processing_time: number;
}
```

#### 2. Health Check

```typescript
GET / health;
Response: {
  status: string;
  ollama_status: string;
  retriever: string;
  generator: string;
}
```

#### 3. System Statistics

```typescript
GET / stats;
Response: {
  qdrant: {
    collection: string;
    points_count: number;
    vector_size: number;
    status: string;
  }
  postgres: {
    total_files: number;
    processed_files: number;
    total_chunks: number;
    avg_tokens_per_chunk: number;
  }
  embedding_model: string;
}
```

#### 4. Model Information

```typescript
GET / model;
Response: {
  data: Array<{
    id: string;
    object: string;
    created: number;
    owned_by: string;
  }>;
}
```

## 🎨 User Interface Design

### Main Chat Interface

```
┌─────────────────────────────────────────────────────────────┐
│ 🏠 RAG Chat System                     ⚙️ Settings  📊 Stats │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  👤 What is machine learning?                               │
│                                                             │
│  🤖 Machine learning is a subset of artificial             │
│     intelligence that enables computers to learn...         │
│                                                             │
│     📚 Sources (3):                                         │
│     • ai-framework.pdf (relevance: 0.87)                   │
│     • ml-guide.pdf (relevance: 0.73)                       │
│     • research.pdf (relevance: 0.65)                       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ 💬 Ask a question...                              [Send] 📤 │
│ ⚙️ top_k: 5  📊 min_score: 0.3  🌡️ temp: 0.1              │
└─────────────────────────────────────────────────────────────┘
```

### Dashboard View

```
┌─────────────────────────────────────────────────────────────┐
│ 📊 System Dashboard                                          │
├─────────────────────────────────────────────────────────────┤
│ ✅ System Health                                            │
│ • Ollama: Healthy                                           │
│ • Retriever: Ready                                          │
│ • Generator: Ready                                          │
│                                                             │
│ 📈 Statistics                                               │
│ • Documents: 25 processed                                   │
│ • Chunks: 1,542 vectors                                     │
│ • Model: llama3.1:8b-instruct-q4_0                         │
│                                                             │
│ 📊 Performance                                              │
│ • Avg Query Time: 2.3s                                     │
│ • Success Rate: 98.5%                                      │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Implementation Phases

### Phase 1: MVP (Days 1-2)

**Core Chat Functionality**

- [ ] Basic React + TypeScript setup
- [ ] Chat interface with message bubbles
- [ ] Query input with send button
- [ ] API integration for /query endpoint
- [ ] Loading states and error handling
- [ ] Source display with scores

### Phase 2: Enhanced UX (Days 3-4)

**Improved User Experience**

- [ ] Settings panel for query parameters
- [ ] Health status indicator
- [ ] Query history/conversation memory
- [ ] Responsive design for mobile
- [ ] Better loading animations
- [ ] Error boundaries

### Phase 3: Dashboard (Days 5-6)

**System Monitoring**

- [ ] Dashboard page with stats
- [ ] Real-time health monitoring
- [ ] Model information display
- [ ] Performance metrics
- [ ] Charts and visualizations

### Phase 4: Production (Days 7-8)

**Production Ready**

- [ ] Multi-stage Docker build
- [ ] Nginx configuration
- [ ] Environment variable management
- [ ] Performance optimization
- [ ] Error logging
- [ ] Accessibility improvements

## 🐳 Docker Configuration

### Development Dockerfile

```dockerfile
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Start development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
```

### Production Dockerfile

```dockerfile
# Build stage
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose Integration

```yaml
frontend:
  build: ./frontend
  container_name: rag-frontend
  ports:
    - "3000:3000"
  environment:
    - REACT_APP_API_URL=http://localhost:8080
    - REACT_APP_ENVIRONMENT=development
  depends_on:
    - rag-api
  volumes:
    - ./frontend:/app
    - /app/node_modules
```

## ⚙️ Configuration Files

### package.json Dependencies

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-query": "^3.39.3",
    "axios": "^1.6.0",
    "lucide-react": "^0.294.0",
    "recharts": "^2.8.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.1.0",
    "typescript": "^5.0.0",
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "vite": "^5.0.0"
  }
}
```

### Environment Variables

```bash
# .env.development
REACT_APP_API_URL=http://localhost:8080
REACT_APP_ENVIRONMENT=development
REACT_APP_APP_NAME="RAG Chat System"

# .env.production
REACT_APP_API_URL=http://localhost:8080
REACT_APP_ENVIRONMENT=production
REACT_APP_APP_NAME="RAG Chat System"
```

## 🔧 Development Workflow

### Setup Commands

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Docker Development

```bash
# Build development container
docker compose build frontend

# Start with hot reload
docker compose up frontend

# Access at http://localhost:3000
```

## 📱 Features Implementation Details

### 1. Chat Interface

- **Message History:** Store conversations in React state
- **Typing Indicators:** Show loading state during API calls
- **Auto-scroll:** Keep latest messages visible
- **Message Formatting:** Markdown support for rich text

### 2. Source Display

- **Expandable Sources:** Click to view full document content
- **Relevance Scores:** Visual indicators (progress bars)
- **File Types:** Icons for different document types
- **Source Filtering:** Sort by relevance/filename

### 3. Settings Panel

- **Query Parameters:** Sliders for top_k, min_score, temperature
- **Real-time Updates:** Apply settings to next query
- **Presets:** Save common configurations
- **Reset Options:** Return to defaults

### 4. Dashboard Analytics

- **Health Monitoring:** Real-time status checks
- **Performance Metrics:** Query time, success rates
- **System Information:** Model details, collection stats
- **Usage Statistics:** Query count, popular topics

## 🚨 Error Handling

### API Error States

- **Network Errors:** Offline indicator with retry
- **Server Errors:** User-friendly error messages
- **Validation Errors:** Form field highlighting
- **Timeout Errors:** Configurable timeout handling

### Loading States

- **Query Processing:** Spinner with progress indication
- **Initial Load:** Skeleton components
- **Background Updates:** Subtle loading indicators
- **Failed States:** Retry buttons with error details

## 🎯 Success Metrics

### Technical Goals

- [ ] **Performance:** < 100ms initial load
- [ ] **Responsiveness:** Works on mobile devices
- [ ] **Reliability:** 99%+ uptime with error handling
- [ ] **Accessibility:** WCAG 2.1 compliance

### User Experience Goals

- [ ] **Intuitive:** New users can query without instructions
- [ ] **Informative:** Clear source attribution and relevance
- [ ] **Responsive:** Real-time feedback and status updates
- [ ] **Professional:** Clean, modern design suitable for demos

## 🚀 Getting Started

1. **Create the frontend directory structure**
2. **Initialize React + TypeScript project with Vite**
3. **Set up TailwindCSS for styling**
4. **Create basic chat interface**
5. **Integrate with RAG API endpoints**
6. **Add Docker configuration**
7. **Test full integration with existing services**

This frontend will transform your RAG system from a technical demonstration into a complete, user-friendly application suitable for presentations, demos, and actual end-user interaction.
