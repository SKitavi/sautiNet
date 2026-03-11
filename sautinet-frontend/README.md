# SautiNet Frontend Dashboard

Simple vanilla JavaScript dashboard for the SautiNet sentiment analysis platform.

## Features

- **Real-time Sentiment Feed**: WebSocket connection showing live analysis results
- **Text Analysis**: Analyze custom text in English, Swahili, or Sheng
- **County Sentiment Map**: View sentiment scores across all 47 Kenyan counties
- **Trending Topics**: See what topics are currently trending
- **Statistics Dashboard**: Track total analyses and sentiment distribution
- **Node Information**: Monitor backend node status and health

## Quick Start

### Option 1: Simple HTTP Server (Python)

```bash
cd sautinet-frontend
python3 -m http.server 3000
```

Then open: http://localhost:3000

### Option 2: Node.js HTTP Server

```bash
cd sautinet-frontend
npx http-server -p 3000
```

Then open: http://localhost:3000

### Option 3: VS Code Live Server

1. Install "Live Server" extension in VS Code
2. Right-click `index.html`
3. Select "Open with Live Server"

## Configuration

The frontend connects to the backend API at `http://localhost:8000` by default.

To change the API endpoint, edit `app.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/feed';
```

## Backend Setup

Make sure the SautiNet backend is running:

```bash
cd ../sautinet-ml-backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

1. **Analyze Text**: Enter text in the input box and click "Analyze"
2. **Live Feed**: Watch real-time sentiment analysis results stream in
3. **County Search**: Filter counties by name in the search box
4. **Pause Feed**: Click "Pause" to stop the live feed temporarily

## Browser Support

Works in all modern browsers with WebSocket support:
- Chrome/Edge 88+
- Firefox 85+
- Safari 14+

## Project Structure

```
sautinet-frontend/
├── index.html      # Main HTML structure
├── styles.css      # All styling (dark theme)
├── app.js          # Application logic & API calls
└── README.md       # This file
```

## API Endpoints Used

- `GET /` - Node information
- `GET /api/v1/health` - Health check
- `GET /api/v1/stats` - Statistics
- `GET /api/v1/counties` - County sentiment data
- `GET /api/v1/trending` - Trending topics
- `POST /api/v1/analyze` - Analyze text
- `WS /ws/feed` - Live sentiment feed

## Troubleshooting

**WebSocket not connecting?**
- Ensure backend is running on port 8000
- Check browser console for errors
- Verify CORS is enabled in backend

**No data showing?**
- Check that backend API is accessible
- Open browser DevTools Network tab
- Verify API responses are successful

**Styling issues?**
- Clear browser cache
- Hard refresh (Ctrl+Shift+R or Cmd+Shift+R)
