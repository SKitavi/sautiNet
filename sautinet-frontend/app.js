// Configuration
const API_BASE_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/feed';

// State
let ws = null;
let feedPaused = false;
let stats = {
    total: 0,
    positive: 0,
    neutral: 0,
    negative: 0
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    // Check API health
    await checkHealth();
    
    // Load initial data
    await loadNodeInfo();
    await loadCounties();
    await loadTrending();
    await loadStats();
    
    // Setup WebSocket
    connectWebSocket();
    
    // Setup event listeners
    setupEventListeners();
    
    // Refresh data periodically
    setInterval(loadStats, 30000); // Every 30s
    setInterval(loadTrending, 60000); // Every 60s
}

// Health Check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/health`);
        const data = await response.json();
        
        updateStatus(true, 'Online');
        return data;
    } catch (error) {
        console.error('Health check failed:', error);
        updateStatus(false, 'Offline');
        return null;
    }
}

function updateStatus(online, text) {
    const indicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    
    indicator.className = `status-indicator ${online ? 'online' : 'offline'}`;
    statusText.textContent = text;
}

// Node Information
async function loadNodeInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        const data = await response.json();
        
        document.getElementById('nodeId').textContent = data.node || 'N/A';
        document.getElementById('nodeRegion').textContent = data.node?.split('-')[0] || 'N/A';
        document.getElementById('nodeVersion').textContent = data.version || 'N/A';
        document.getElementById('nodeHealthStatus').textContent = data.status || 'N/A';
    } catch (error) {
        console.error('Failed to load node info:', error);
    }
}

// Statistics
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/stats`);
        const data = await response.json();
        
        if (data.total_processed !== undefined) {
            stats.total = data.total_processed;
            stats.positive = data.sentiment_distribution?.positive || 0;
            stats.neutral = data.sentiment_distribution?.neutral || 0;
            stats.negative = data.sentiment_distribution?.negative || 0;
            
            updateStatsDisplay();
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

function updateStatsDisplay() {
    document.getElementById('statTotal').textContent = stats.total;
    document.getElementById('statPositive').textContent = stats.positive;
    document.getElementById('statNeutral').textContent = stats.neutral;
    document.getElementById('statNegative').textContent = stats.negative;
}

// Counties
async function loadCounties() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/counties`);
        const data = await response.json();
        
        displayCounties(data.counties || []);
    } catch (error) {
        console.error('Failed to load counties:', error);
        document.getElementById('countyList').innerHTML = '<p class="loading">Failed to load counties</p>';
    }
}

function displayCounties(counties) {
    const container = document.getElementById('countyList');
    
    if (!counties || counties.length === 0) {
        container.innerHTML = '<p class="loading">No county data available</p>';
        return;
    }
    
    container.innerHTML = counties.map(county => {
        const sentiment = county.avg_sentiment || 0;
        const sentimentClass = sentiment > 0.1 ? 'positive' : sentiment < -0.1 ? 'negative' : 'neutral';
        const barWidth = Math.abs(sentiment) * 100;
        
        return `
            <div class="county-item">
                <span class="county-name">${county.name}</span>
                <div class="county-sentiment">
                    <span class="sentiment-score">${sentiment.toFixed(2)}</span>
                    <div class="sentiment-bar">
                        <div class="sentiment-fill ${sentimentClass}" style="width: ${barWidth}%"></div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// Trending Topics
async function loadTrending() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/trending`);
        const data = await response.json();
        
        displayTrending(data.topics || []);
    } catch (error) {
        console.error('Failed to load trending:', error);
        document.getElementById('trendingTopics').innerHTML = '<p class="loading">Failed to load trending topics</p>';
    }
}

function displayTrending(topics) {
    const container = document.getElementById('trendingTopics');
    
    if (!topics || topics.length === 0) {
        container.innerHTML = '<p class="loading">No trending topics available</p>';
        return;
    }
    
    container.innerHTML = topics.slice(0, 10).map(topic => `
        <div class="topic-item">
            <span class="topic-name">${topic.topic || topic.name}</span>
            <span class="topic-count">${topic.count || topic.mentions || 0}</span>
        </div>
    `).join('');
}

// Text Analysis
async function analyzeText(text) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text })
        });
        
        const data = await response.json();
        displayAnalysisResult(data);
    } catch (error) {
        console.error('Analysis failed:', error);
        alert('Failed to analyze text. Please try again.');
    }
}

function displayAnalysisResult(data) {
    const container = document.getElementById('analysisResult');
    container.style.display = 'block';
    
    // Language
    const langMap = { 'en': 'English', 'sw': 'Swahili', 'sh': 'Sheng' };
    const language = data.language?.detected_language || 'unknown';
    const confidence = data.language?.confidence || 0;
    document.getElementById('resultLanguage').textContent = 
        `${langMap[language] || language} (${(confidence * 100).toFixed(0)}%)`;
    
    // Sentiment
    const sentiment = data.sentiment?.label || 'neutral';
    const sentimentScore = data.sentiment?.score || 0;
    const sentimentEl = document.getElementById('resultSentiment');
    sentimentEl.textContent = `${sentiment} (${sentimentScore.toFixed(2)})`;
    sentimentEl.className = `result-value sentiment-badge ${sentiment}`;
    
    // Confidence
    const sentimentConfidence = data.sentiment?.confidence || 0;
    document.getElementById('resultConfidence').textContent = 
        `${(sentimentConfidence * 100).toFixed(0)}%`;
    
    // Topic
    const topic = data.topics?.primary_topic || 'general';
    const isPolitical = data.topics?.is_political ? '🏛️ Political' : '';
    document.getElementById('resultTopic').textContent = `${topic} ${isPolitical}`;
    
    // Entities
    const entitiesContainer = document.getElementById('resultEntities');
    if (data.entities && Object.keys(data.entities).length > 0) {
        const entityTags = [];
        for (const [type, items] of Object.entries(data.entities)) {
            if (items && items.length > 0) {
                items.forEach(item => {
                    entityTags.push(`<span class="entity-tag">${type}: ${item}</span>`);
                });
            }
        }
        if (entityTags.length > 0) {
            entitiesContainer.innerHTML = `<strong>Entities:</strong><br>${entityTags.join('')}`;
            entitiesContainer.style.display = 'block';
        } else {
            entitiesContainer.style.display = 'none';
        }
    } else {
        entitiesContainer.style.display = 'none';
    }
    
    // Sheng indicators
    const shengContainer = document.getElementById('resultSheng');
    if (data.language?.sheng_indicators && data.language.sheng_indicators.length > 0) {
        const shengTags = data.language.sheng_indicators.map(word => 
            `<span class="sheng-tag">${word}</span>`
        ).join('');
        shengContainer.innerHTML = `<strong>Sheng words detected:</strong><br>${shengTags}`;
        shengContainer.style.display = 'block';
    } else {
        shengContainer.style.display = 'none';
    }
}

// WebSocket for Live Feed
function connectWebSocket() {
    try {
        ws = new WebSocket(WS_URL);
        
        ws.onopen = () => {
            console.log('WebSocket connected');
            updateStatus(true, 'Live');
        };
        
        ws.onmessage = (event) => {
            if (!feedPaused) {
                const data = JSON.parse(event.data);
                addFeedItem(data);
                updateStatsFromFeed(data);
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        ws.onclose = () => {
            console.log('WebSocket disconnected');
            updateStatus(false, 'Disconnected');
            
            // Reconnect after 5 seconds
            setTimeout(connectWebSocket, 5000);
        };
    } catch (error) {
        console.error('Failed to connect WebSocket:', error);
    }
}

function addFeedItem(data) {
    const container = document.getElementById('liveFeed');
    
    // Remove placeholder
    const placeholder = container.querySelector('.feed-placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    const sentiment = data.sentiment?.label || 'neutral';
    const text = data.text || data.message || 'No text';
    const language = data.language?.detected_language || 'unknown';
    const topic = data.topics?.primary_topic || 'general';
    
    const feedItem = document.createElement('div');
    feedItem.className = `feed-item ${sentiment}`;
    feedItem.innerHTML = `
        <div class="feed-text">${escapeHtml(text.substring(0, 150))}${text.length > 150 ? '...' : ''}</div>
        <div class="feed-meta">
            <span>Lang: ${language.toUpperCase()}</span>
            <span>Sentiment: ${sentiment}</span>
            <span>Topic: ${topic}</span>
        </div>
    `;
    
    container.insertBefore(feedItem, container.firstChild);
    
    // Keep only last 20 items
    while (container.children.length > 20) {
        container.removeChild(container.lastChild);
    }
}

function updateStatsFromFeed(data) {
    stats.total++;
    
    const sentiment = data.sentiment?.label || 'neutral';
    if (sentiment === 'positive') stats.positive++;
    else if (sentiment === 'negative') stats.negative++;
    else stats.neutral++;
    
    updateStatsDisplay();
}

// Event Listeners
function setupEventListeners() {
    // Analyze form
    document.getElementById('analyzeForm').addEventListener('submit', (e) => {
        e.preventDefault();
        const text = document.getElementById('textInput').value.trim();
        if (text) {
            analyzeText(text);
        }
    });
    
    // Toggle feed
    document.getElementById('toggleFeed').addEventListener('click', () => {
        feedPaused = !feedPaused;
        const btn = document.getElementById('toggleFeed');
        btn.textContent = feedPaused ? 'Resume' : 'Pause';
    });
    
    // County search
    document.getElementById('countySearch').addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        const countyItems = document.querySelectorAll('.county-item');
        
        countyItems.forEach(item => {
            const countyName = item.querySelector('.county-name').textContent.toLowerCase();
            item.style.display = countyName.includes(searchTerm) ? 'flex' : 'none';
        });
    });
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
