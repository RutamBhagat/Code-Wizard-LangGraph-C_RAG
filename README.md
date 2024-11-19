# Code Wizard: LangChain Documentation AI Chatbot

## Overview

Code Wizard is an AI-powered chatbot designed to make understanding and using LangChain documentation effortless. Ask any question related to LangChain concepts or code, and Code Wizard will explain it clearly and interactively. It is built using a tech stack that includes Next.js, FastAPI, LangChain, LangGraph, and LCEL, with the ability to switch between models like ChatOpenAI and local LLaMA models.

**Frontend:** [Code Wizard UI](https://code-wizard-frontend.vercel.app/)

**Frontend Repo:** [Code Wizard Frontend](https://github.com/RutamBhagat/code_wizard_frontend)

**Backend Repo:** [Code Wizard Backend](https://github.com/RutamBhagat/Code-Wizard-LangGraph-C_RAG)

## Demo Video

https://github.com/user-attachments/assets/399aae3e-a1fc-4160-b878-4aa43cd28a38

## Key Features

- **Interactive Chat Interface**: Engaging chat interface built with Next.js and React for smooth and intuitive user experience
- **LangChain Integration**: Uses LangChain for building applications with large language models
- **Documentation Search**: Implements LangGraph DAG to search vector databases for relevant documentation chunks
- **Custom AI Responses**: Combines retrieved documentation chunks with ChatOpenAI to generate detailed answers
- **Markdown Rendering**: Supports rendering code snippets and Markdown for easy comprehension

## Technologies Used

- **Frontend**: Next.js, Typescript for a responsive and dynamic user interface
- **Backend**: FastAPI for fast and reliable API handling
- **AI Frameworks**: LangChain, LangGraph, LCEL for processing and understanding queries
- **Model Support**: Switchable between ChatOpenAI and LLaMA models for flexibility
- **Data Storage**: Vector databases for efficient document retrieval

## Challenges and Learnings

Building Code Wizard was a fantastic learning journey, offering valuable lessons on:

- **LangChain Mastery**: Leveraging components like agents, memory, and vector stores effectively
- **Model Optimization**: Techniques like quantization and CPU offloading for efficient performance
- **UI/UX Design**: Creating conversational interfaces that feel natural and easy to use
- **Scalable Backend Architecture**: Using FastAPI and async processing for better performance

## Optimizations

1. **Caching System**

   - Cached responses for frequently asked questions to improve latency and efficiency
   - Reduced API load and provided faster user experiences

2. **Streaming Responses**

   - Implemented LangChain’s streaming feature to send data to users as soon as it’s available
   - Enhanced user interaction by reducing waiting times

3. **Model Flexibility**
   - Capability to switch to more powerful models like GPT-4 for critical use cases
   - Balances performance and cost-effectiveness based on user needs

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+
- Docker (optional)
- Git

### Frontend Setup

```bash
# Clone repository
git clone https://github.com/RutamBhagat/code_wizard_frontend
cd code_wizard_frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend Setup

```bash
   # Backend
   git clone https://github.com/RutamBhagat/Code-Wizard-LangGraph-C_RAG

   pipx install pdm

   pdm install

   source .venv/bin/activate

   pdm run uvicorn app.server:app --reload
```

#### If you want to setup using Docker

```bash
   # Remove the old container if present
   docker rm code-wizard-container

   # Build the new image with no cache
   docker build --no-cache -t code-wizard-app .

   # Run the container
   docker run -d -p 8000:8000 --name code-wizard-container code-wizard-app
```

2. **Configure Environment**

   - Set up necessary API keys and configurations for LangChain and models
   - Adjust settings for vector databases and data storage

3. **Install Dependencies**
   - Follow setup instructions in the repos to install dependencies
   - Use Python and npm to ensure the backend and frontend are configured properly

## Outcome

Code Wizard has demonstrated its ability to transform the way developers learn and utilize the LangChain framework. It offers seamless integration of documentation search and AI-based explanations while being highly optimized for scalability and performance.

# Screenshots

![Screenshot 1](https://github.com/user-attachments/assets/85720540-b534-4907-bfe3-da125306e684)
![Screenshot 2](https://github.com/user-attachments/assets/f798dff9-eae2-4818-b0f9-620ea596e034)
![Screenshot 3](https://github.com/user-attachments/assets/288af3ed-bc70-4191-9bf6-b00c8f44fdd2)
![Screenshot 4](https://github.com/user-attachments/assets/e31b8075-d5ef-4a28-a113-37fc9f8c2842)
