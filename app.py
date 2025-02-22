# app.py

import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
import uuid
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from collections import defaultdict, Counter
import networkx as nx
from flask import Flask, render_template, request, jsonify, session
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = os.urandom(24)

# SQLite database setup
DB_FILE = "chat_history.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (user_id TEXT, role TEXT, content TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()
    logging.info("Initialized chat history database")


init_db()


class Config:
    HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    STM_FILE = "short_term_memory.json"
    LTM_FILE = "long_term_memory.json"
    SEMANTIC_FILE = "semantic_memory.json"
    GRAPH_FILE = "memory_graph.json"
    FAISS_INDEX_FILE = "faiss_index.bin"

    STM_CAPACITY = 50
    STM_RETENTION_HOURS = 24
    BASE_CONSOLIDATION_THRESHOLD = 0.7
    MAX_RETRIEVED_MEMORIES = 5
    FORGETTING_RATE = 0.05
    EMBEDDING_DIM = 768


class TransformerEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.embedding_dim = Config.EMBEDDING_DIM
            logging.info(f"Initialized TransformerEmbedder with {model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize TransformerEmbedder: {e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        try:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings
        except Exception as e:
            logging.error(f"Error encoding texts: {e}")
            return np.zeros((len(texts), self.embedding_dim))

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@dataclass
class Memory:
    content: str
    embedding: np.ndarray
    timestamp: str
    importance_score: float
    memory_type: str
    context_tags: List[str]
    related_memories: List[str]
    emotional_valence: float
    access_count: int = 0
    strength: float = 1.0
    episode_id: Optional[str] = None

    def to_dict(self):
        return {
            "content": self.content,
            "embedding": self.embedding.tolist(),
            "timestamp": self.timestamp,
            "importance_score": self.importance_score,
            "memory_type": self.memory_type,
            "context_tags": self.context_tags,
            "related_memories": self.related_memories,
            "emotional_valence": self.emotional_valence,
            "access_count": self.access_count,
            "strength": self.strength,
            "episode_id": self.episode_id
        }

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                content=data["content"],
                embedding=np.array(data["embedding"]),
                timestamp=data["timestamp"],
                importance_score=data["importance_score"],
                memory_type=data["memory_type"],
                context_tags=data["context_tags"],
                related_memories=data["related_memories"],
                emotional_valence=data["emotional_valence"],
                access_count=data.get("access_count", 0),
                strength=data.get("strength", 1.0),
                episode_id=data.get("episode_id")
            )
        except Exception as e:
            logging.error(f"Error creating Memory from dict: {e}")
            raise


class EnhancedMemorySystem:
    def __init__(self):
        try:
            self.embedder = TransformerEmbedder()
            self.embedding_dim = Config.EMBEDDING_DIM
            self.short_term_memory: List[Memory] = []
            self.long_term_memory: List[Memory] = []
            self.semantic_memory: Dict[str, List[str]] = defaultdict(list)
            self.memory_graph = nx.DiGraph()
            self.episode_tracker = defaultdict(list)
            try:
                from transformers import pipeline
                self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                logging.info("Initialized summarizer")
            except Exception as e:
                logging.warning(f"Failed to initialize summarizer: {e}. Proceeding without summarization.")
                self.summarizer = None

            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.memory_id_to_content = {}
            self.next_id = 0

            self.load_memories()
            self.rebuild_faiss_index()
            logging.info("EnhancedMemorySystem initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize EnhancedMemorySystem: {e}")
            raise

    def extract_context_tags(self, text: str) -> List[str]:
        common_topics = ["work", "family", "health", "technology", "education"]
        return [topic for topic in common_topics if topic.lower() in text.lower()]

    def analyze_emotion(self, text: str) -> float:
        positive_words = {"happy", "good", "great", "excellent", "awesome"}
        negative_words = {"sad", "bad", "poor", "terrible", "awful"}
        score = sum(1 for word in text.lower().split() if word in positive_words) - \
                sum(1 for word in text.lower().split() if word in negative_words)
        return max(-1, min(1, score / max(len(text.split()), 1)))

    def calculate_importance(self, text: str, context_tags: List[str], emotional_valence: float) -> float:
        length_score = min(1.0, len(text) / 1000)
        context_score = len(context_tags) / 5
        emotion_score = abs(emotional_valence)
        return (length_score + context_score + emotion_score) / 3

    def apply_forgetting_curve(self, memory: Memory):
        days_since = (datetime.now() - datetime.fromisoformat(memory.timestamp)).days
        decay = Config.FORGETTING_RATE * days_since
        memory.strength = max(0.1, memory.strength * (1 - decay) + (memory.access_count * 0.1))

    def compress_memory(self, memory: Memory) -> Memory:
        if memory.strength < 0.3 and len(memory.content) > 100 and self.summarizer:
            try:
                summary = self.summarizer(memory.content, max_length=50, min_length=20, do_sample=False)[0][
                    'summary_text']
                memory.content = f"[Compressed] {summary}"
                memory.embedding = self.embedder.encode_single(memory.content)
            except Exception as e:
                logging.error(f"Failed to compress memory: {e}")
        return memory

    def assign_episode_id(self, memory: Memory, prev_memory: Optional[Memory]):
        if prev_memory and (datetime.fromisoformat(memory.timestamp) -
                            datetime.fromisoformat(prev_memory.timestamp)) < timedelta(minutes=30):
            memory.episode_id = prev_memory.episode_id
        else:
            memory.episode_id = f"EP_{len(self.episode_tracker)}"
        self.episode_tracker[memory.episode_id].append(memory)

    def add_memory(self, text: str, prev_memory: Optional[Memory] = None):
        try:
            embedding = self.embedder.encode_single(text)
            context_tags = self.extract_context_tags(text)
            emotional_valence = self.analyze_emotion(text)
            importance_score = self.calculate_importance(text, context_tags, emotional_valence)

            memory = Memory(
                content=text,
                embedding=embedding,
                timestamp=datetime.now().isoformat(),
                importance_score=importance_score,
                memory_type="short_term",
                context_tags=context_tags,
                related_memories=[],
                emotional_valence=emotional_valence
            )

            self.assign_episode_id(memory, prev_memory)
            self.short_term_memory.append(memory)
            self.update_memory_graph(memory)
            self.add_to_faiss(memory)
            self.consolidate_memories()
            self.update_semantic_memory(memory)
            self.save_memories()
            return memory
        except Exception as e:
            logging.error(f"Error adding memory: {e}")
            raise

    def update_memory_graph(self, memory: Memory):
        self.memory_graph.add_node(memory.content)
        for existing in self.short_term_memory + self.long_term_memory:
            if existing != memory:
                try:
                    similarity = self.calculate_similarity(memory.embedding, existing.embedding)
                    if similarity > 0.6:
                        self.memory_graph.add_edge(memory.content, existing.content, weight=similarity)
                        memory.related_memories.append(existing.content)
                except Exception as e:
                    logging.error(f"Error updating memory graph: {e}")

    def consolidate_memories(self):
        current_time = datetime.now()
        consolidated = []
        total_memories = len(self.short_term_memory) + len(self.long_term_memory)
        adaptive_threshold = Config.BASE_CONSOLIDATION_THRESHOLD * (1 + total_memories / 1000)

        for memory in self.short_term_memory[:]:
            try:
                self.apply_forgetting_curve(memory)
                age = current_time - datetime.fromisoformat(memory.timestamp)
                attentions = self.compute_attention_weights(memory.embedding, [m.embedding for m in
                                                                               self.short_term_memory + self.long_term_memory])
                cluster_score = attentions.mean() if attentions.size > 0 else 0.0

                if (age > timedelta(hours=Config.STM_RETENTION_HOURS) or
                        memory.importance_score + memory.strength + cluster_score >= adaptive_threshold):
                    memory = self.compress_memory(memory)
                    memory.memory_type = "long_term"
                    self.long_term_memory.append(memory)
                    consolidated.append(memory)
            except Exception as e:
                logging.error(f"Error consolidating memory: {e}")

        self.short_term_memory = [m for m in self.short_term_memory if m not in consolidated]

    def update_semantic_memory(self, memory: Memory):
        for tag in memory.context_tags:
            self.semantic_memory[tag].append(memory.content)
            if len(self.semantic_memory[tag]) > Config.MAX_RETRIEVED_MEMORIES:
                self.semantic_memory[tag] = self.semantic_memory[tag][-Config.MAX_RETRIEVED_MEMORIES:]

    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        try:
            return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
        except Exception as e:
            logging.error(f"Error calculating similarity: {e}")
            return 0.0

    def compute_attention_weights(self, query_embedding: np.ndarray, memory_embeddings: List[np.ndarray]) -> np.ndarray:
        if not memory_embeddings:
            return np.array([])
        try:
            embeddings = np.vstack(memory_embeddings)
            scores = np.dot(embeddings, query_embedding)
            attention_weights = np.exp(scores) / (np.sum(np.exp(scores)) + 1e-10)
            return attention_weights
        except Exception as e:
            logging.error(f"Error computing attention weights: {e}")
            return np.zeros(len(memory_embeddings))

    def add_to_faiss(self, memory: Memory):
        try:
            self.faiss_index.add(memory.embedding.reshape(1, -1).astype(np.float32))
            self.memory_id_to_content[self.next_id] = memory.content
            self.next_id += 1
        except Exception as e:
            logging.error(f"Error adding to FAISS: {e}")

    def rebuild_faiss_index(self):
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.memory_id_to_content = {}
        self.next_id = 0
        all_memories = self.short_term_memory + self.long_term_memory
        if all_memories:
            try:
                embeddings = self.embedder.encode([m.content for m in all_memories]).astype(np.float32)
                for i, memory in enumerate(all_memories):
                    memory.embedding = embeddings[i]
                    self.memory_id_to_content[i] = memory.content
                self.faiss_index.add(embeddings)
                self.next_id = len(all_memories)
                logging.info(f"Rebuilt FAISS index with {len(all_memories)} memories")
            except Exception as e:
                logging.error(f"Error rebuilding FAISS index: {e}")

    def retrieve_relevant_memories(self, query: str) -> List[str]:
        try:
            query_embedding = self.embedder.encode_single(query)
            context_tags = self.extract_context_tags(query)
            memory_scores = defaultdict(float)

            all_memories = self.short_term_memory + self.long_term_memory

            if all_memories:
                attention_weights = self.compute_attention_weights(query_embedding, [m.embedding for m in all_memories])
                for memory, weight in zip(all_memories, attention_weights):
                    memory.access_count += 1
                    memory.strength = min(1.0, memory.strength + 0.05)
                    memory_scores[memory.content] += weight * memory.strength * (
                        1.2 if memory in self.short_term_memory else 1.0)

            try:
                D, I = self.faiss_index.search(query_embedding.reshape(1, -1).astype(np.float32),
                                               Config.MAX_RETRIEVED_MEMORIES)
                for dist, idx in zip(D[0], I[0]):
                    if idx != -1 and idx in self.memory_id_to_content:
                        content = self.memory_id_to_content[idx]
                        memory_scores[content] += 1.0 / (dist + 1e-6)
            except Exception as e:
                logging.error(f"Error in FAISS search: {e}")

            for episode_id, memories in self.episode_tracker.items():
                if any(self.calculate_similarity(query_embedding, m.embedding) > 0.5 for m in memories):
                    for m in memories:
                        memory_scores[m.content] += 0.3

            for node in self.memory_graph.nodes:
                memory = next((m for m in all_memories if m.content == node), None)
                if memory and self.calculate_similarity(query_embedding, memory.embedding) > 0.6:
                    for neighbor in self.memory_graph.neighbors(node):
                        memory_scores[neighbor] += 0.2

            for tag in context_tags:
                for content in self.semantic_memory.get(tag, []):
                    memory_scores[content] += 0.25

            sorted_memories = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
            return [content for content, _ in sorted_memories[:Config.MAX_RETRIEVED_MEMORIES]]
        except Exception as e:
            logging.error(f"Error retrieving memories: {e}")
            return []

    def save_memories(self):
        try:
            with open(Config.STM_FILE, 'w') as f:
                json.dump([m.to_dict() for m in self.short_term_memory], f)
            with open(Config.LTM_FILE, 'w') as f:
                json.dump([m.to_dict() for m in self.long_term_memory], f)
            with open(Config.SEMANTIC_FILE, 'w') as f:
                json.dump(dict(self.semantic_memory), f)
            with open(Config.GRAPH_FILE, 'w') as f:
                json.dump(nx.readwrite.json_graph.node_link_data(self.memory_graph, edges="links"), f)
            faiss.write_index(self.faiss_index, Config.FAISS_INDEX_FILE)
            with open("faiss_mapping.json", 'w') as f:
                json.dump(self.memory_id_to_content, f)
            logging.info("Memories saved successfully")
        except Exception as e:
            logging.error(f"Error saving memories: {e}")

    def load_memories(self):
        try:
            if os.path.exists(Config.STM_FILE):
                with open(Config.STM_FILE, 'r') as f:
                    self.short_term_memory = [Memory.from_dict(m) for m in json.load(f)]
            if os.path.exists(Config.LTM_FILE):
                with open(Config.LTM_FILE, 'r') as f:
                    self.long_term_memory = [Memory.from_dict(m) for m in json.load(f)]
            if os.path.exists(Config.SEMANTIC_FILE):
                with open(Config.SEMANTIC_FILE, 'r') as f:
                    self.semantic_memory = defaultdict(list, json.load(f))
            if os.path.exists(Config.GRAPH_FILE):
                try:
                    with open(Config.GRAPH_FILE, 'r') as f:
                        self.memory_graph = nx.readwrite.json_graph.node_link_graph(json.load(f), edges="links")
                except Exception as e:
                    logging.error(f"Error loading memory graph: {e}. Starting with empty graph.")
                    self.memory_graph = nx.DiGraph()
            if os.path.exists(Config.FAISS_INDEX_FILE) and os.path.exists("faiss_mapping.json"):
                self.faiss_index = faiss.read_index(Config.FAISS_INDEX_FILE)
                with open("faiss_mapping.json", 'r') as f:
                    self.memory_id_to_content = json.load(f)
                self.next_id = len(self.memory_id_to_content)
            logging.info("Memories loaded successfully")
        except Exception as e:
            logging.error(f"Error loading memories: {e}")


class Chatbot:
    def __init__(self):
        try:
            self.memory_system = EnhancedMemorySystem()
            from huggingface_hub import InferenceClient
            if not Config.HF_API_TOKEN:
                raise ValueError("HF_API_TOKEN not set in environment")
            self.api = InferenceClient(token=Config.HF_API_TOKEN)
            logging.info("Initialized Hugging Face InferenceClient")
        except Exception as e:
            logging.error(f"Failed to initialize Chatbot: {e}")
            self.api = None
        self.last_memory = None

    def generate_response(self, user_input: str, user_id: str) -> str:
        try:
            if not self.api:
                raise ValueError("InferenceClient not initialized")
            relevant_memories = self.memory_system.retrieve_relevant_memories(user_input)
            context = "Relevant memories:\n" + "\n".join(f"{i + 1}. {m}" for i, m in enumerate(relevant_memories)) \
                if relevant_memories else ""

            prompt = f"""{context}

Current user message: {user_input}

Respond naturally, incorporating relevant memories if applicable."""

            response_text = self.api.text_generation(
                prompt,
                max_new_tokens=350,
                model="meta-llama/Llama-3.2-11B-Vision-Instruct",
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.5
            )
            full_memory = f"User: {user_input}\nAssistant: {response_text}"
            self.last_memory = self.memory_system.add_memory(full_memory, self.last_memory)

            timestamp = datetime.now().isoformat()
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                      (user_id, "user", user_input, timestamp))
            c.execute("INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                      (user_id, "assistant", response_text, timestamp))
            conn.commit()
            conn.close()

            logging.info(f"Generated response for input: {user_input} by user {user_id}")
            return response_text
        except Exception as e:
            logging.error(f"API error in generate_response: {e}")
            return f"Sorry, I encountered an error: {str(e)}. Try again later."

    def get_chat_history(self, user_id: str) -> List[Dict[str, str]]:
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT role, content FROM messages WHERE user_id = ? ORDER BY timestamp", (user_id,))
            history = [{"role": row[0], "content": row[1]} for row in c.fetchall()]
            conn.close()
            logging.info(f"Retrieved chat history for user {user_id}")
            return history
        except Exception as e:
            logging.error(f"Error retrieving chat history for user {user_id}: {e}")
            return []


chatbot = Chatbot()


@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    user_id = session['user_id']
    messages = chatbot.get_chat_history(user_id)
    return render_template('index.html', messages=messages)


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    user_id = session.get('user_id', str(uuid.uuid4()))
    session['user_id'] = user_id

    response = chatbot.generate_response(user_input, user_id)
    return jsonify({'response': response})


@app.route('/stats')
def stats():
    ms = chatbot.memory_system
    stats = {
        'short_term': len(ms.short_term_memory),
        'long_term': len(ms.long_term_memory),
        'episodes': len(ms.episode_tracker)
    }
    return jsonify(stats)


@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)