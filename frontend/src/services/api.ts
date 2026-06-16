const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  async chat(question: string, history: [string, string][], signal?: AbortSignal) {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        history,
        translate: true,
      }),
      signal,
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => 'Unknown error');
      throw new Error(`Chat request failed (${response.status}): ${detail}`);
    }

    return response.json();
  },

  async uploadCase(file: File, onProgress?: (progress: number) => void) {
    const formData = new FormData();
    formData.append('file', file);

    return new Promise<{ summary: string; chunks: number; status: string }>((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = Math.round((event.loaded / event.total) * 100);
          onProgress(progress);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            reject(new Error('Invalid response from server'));
          }
        } else {
          reject(new Error(`Server error: ${xhr.statusText}`));
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error during upload'));
      });

      xhr.open('POST', `${API_BASE_URL}/upload`);
      xhr.send(formData);
    });
  },

  async clearCase() {
    const response = await fetch(`${API_BASE_URL}/case`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Failed to clear case document');
    return response.json();
  },

  async getCaseStatus() {
    const response = await fetch(`${API_BASE_URL}/case`);
    if (!response.ok) throw new Error('Failed to get case status');
    return response.json();
  },
};
