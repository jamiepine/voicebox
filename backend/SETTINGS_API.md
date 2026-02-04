# Settings API Usage Example

The voicebox backend now has settings stored in the database that can be changed via the UI without restarting the app.

## Get All Settings

```bash
GET /api/settings
```

Response:
```json
{
  "settings": {
    "tts_backend": "chatterbox_turbo"
  }
}
```

## Get Backend Options

```bash
GET /api/settings/backend/options
```

Response:
```json
{
  "options": [
    {
      "value": "chatterbox_turbo",
      "label": "Chatterbox Turbo",
      "description": "Fast, high-quality TTS (4GB download on first use, uses MPS on Apple Silicon)"
    },
    {
      "value": "qwen",
      "label": "Qwen TTS",
      "description": "Alternative TTS model with MLX support"
    }
  ]
}
```

## Update Backend Setting

```bash
PUT /api/settings/tts_backend
Content-Type: application/json

{
  "value": "qwen"
}
```

Response:
```json
{
  "key": "tts_backend",
  "value": "qwen",
  "reload_required": true
}
```

## Frontend Integration

In your Tauri frontend, you can now:

1. **Show backend selector** in settings UI
2. **Fetch current backend** on app load
3. **Switch backends** without restart
4. **Display backend info** in settings

Example TypeScript/React code:

```typescript
// Fetch available options
const response = await fetch('http://localhost:8000/api/settings/backend/options');
const { options } = await response.json();

// Get current backend
const settingsResponse = await fetch('http://localhost:8000/api/settings/tts_backend');
const { value: currentBackend } = await settingsResponse.json();

// Update backend
const updateResponse = await fetch('http://localhost:8000/api/settings/tts_backend', {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ value: 'chatterbox_turbo' })
});
const result = await updateResponse.json();
console.log('Backend updated:', result.value);
```

## How It Works

1. **Database-backed**: Settings are stored in a `settings` table
2. **Dynamic loading**: Backend checks the database on initialization
3. **Hot reload**: Changing the backend reloads it without restart
4. **Fallback**: If DB isn't initialized, falls back to `TTS_MODEL_TYPE` env var

This means no more environment variables needed - everything is controlled through the UI!
