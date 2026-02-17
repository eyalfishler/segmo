const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5050;
const DATA_DIR = path.join(__dirname, 'data');

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR);

app.use(cors());
app.use(express.json({ limit: '5mb' }));

// Serve viewer
app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'viewer.html'));
});

// Sanitize clientId to a safe filename
function safeFilename(clientId) {
  if (!clientId) return '_anonymous';
  return clientId.replace(/[^a-zA-Z0-9_\-]/g, '_').slice(0, 100);
}

// Append an event to a client's JSONL file
function appendEvent(clientId, event) {
  const filename = safeFilename(clientId) + '.jsonl';
  const filepath = path.join(DATA_DIR, filename);
  const line = JSON.stringify({ receivedAt: Date.now(), ...event }) + '\n';
  fs.appendFileSync(filepath, line);
}

// POST /api/telemetry — receive DiagnosticEvent
app.post('/api/telemetry', (req, res) => {
  const event = req.body;
  const clientId = event.clientId || null;
  appendEvent(clientId, event);
  console.log(`[telemetry] ${event.type} from ${clientId || 'anonymous'}`);
  res.json({ ok: true });
});

// POST /api/telemetry/snapshot — receive DiagnosticSnapshot
app.post('/api/telemetry/snapshot', (req, res) => {
  const snapshot = req.body;
  const clientId = snapshot.clientId || null;
  appendEvent(clientId, { type: 'snapshot', data: snapshot, timestamp: Date.now(), clientId });
  console.log(`[telemetry] snapshot from ${clientId || 'anonymous'}`);
  res.json({ ok: true });
});

// GET /api/telemetry/clients — list all clients
app.get('/api/telemetry/clients', (_req, res) => {
  const files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.jsonl'));
  const clients = files.map(f => {
    const id = f.replace('.jsonl', '');
    const stat = fs.statSync(path.join(DATA_DIR, f));
    return {
      id,
      lastEvent: stat.mtimeMs,
      sizeBytes: stat.size,
    };
  });
  clients.sort((a, b) => b.lastEvent - a.lastEvent);
  res.json(clients);
});

// GET /api/telemetry/clients/:id — get all events for a client
app.get('/api/telemetry/clients/:id', (req, res) => {
  const filename = safeFilename(req.params.id) + '.jsonl';
  const filepath = path.join(DATA_DIR, filename);

  if (!fs.existsSync(filepath)) {
    return res.status(404).json({ error: 'Client not found' });
  }

  const lines = fs.readFileSync(filepath, 'utf-8').trim().split('\n');
  const events = lines.filter(Boolean).map(line => JSON.parse(line));
  res.json(events);
});

// DELETE /api/telemetry/clients/:id — delete a client's data
app.delete('/api/telemetry/clients/:id', (req, res) => {
  const filename = safeFilename(req.params.id) + '.jsonl';
  const filepath = path.join(DATA_DIR, filename);

  if (fs.existsSync(filepath)) {
    fs.unlinkSync(filepath);
  }
  res.json({ ok: true });
});

app.listen(PORT, () => {
  console.log(`Telemetry server running at http://localhost:${PORT}`);
  console.log(`Data directory: ${DATA_DIR}`);
});
