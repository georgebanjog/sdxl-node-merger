/* ═══════════════════════════════════════════════════════════════
   API — HTTP + WebSocket communication with backend
   ═══════════════════════════════════════════════════════════════ */

window.Api = (() => {
    let ws = null;
    let wsReconnectTimer = null;
    const WS_RECONNECT_INTERVAL = 3000;

    function init() {
        connectWebSocket();
    }

    // ─── HTTP ───────────────────────────────────────────────────
    async function get(url) {
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (e) {
            console.error('API GET error:', url, e);
            return null;
        }
    }

    async function post(url, data = {}) {
        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (e) {
            console.error('API POST error:', url, e);
            return null;
        }
    }

    // ─── WebSocket ──────────────────────────────────────────────
    function connectWebSocket() {
        const host = location.hostname || '127.0.0.1';
        const port = parseInt(location.port || '8765') + 1; // WS port = HTTP port + 1

        try {
            ws = new WebSocket(`ws://${host}:${port}`);

            ws.onopen = () => {
                console.log('WebSocket connected');
                updateConnectionStatus(true);
                if (wsReconnectTimer) {
                    clearInterval(wsReconnectTimer);
                    wsReconnectTimer = null;
                }
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleWSMessage(data);
                } catch (e) {
                    console.error('WS message parse error:', e);
                }
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                updateConnectionStatus(false);
                scheduleReconnect();
            };

            ws.onerror = (err) => {
                console.error('WebSocket error:', err);
                updateConnectionStatus(false);
            };
        } catch (e) {
            console.error('WebSocket connection failed:', e);
            scheduleReconnect();
        }
    }

    function scheduleReconnect() {
        if (!wsReconnectTimer) {
            wsReconnectTimer = setInterval(() => {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    connectWebSocket();
                }
            }, WS_RECONNECT_INTERVAL);
        }
    }

    function handleWSMessage(data) {
        switch (data.type) {
            case 'connected':
                console.log('Server:', data.message);
                break;

            case 'progress':
                updateProgress(data);
                break;

            case 'execution_start':
                if (window.showToast) showToast(data.message || 'Merge started', 'info');
                break;

            case 'execution_complete':
                onExecutionComplete(data.result);
                break;

            case 'execution_error':
                onExecutionError(data.error);
                break;

            case 'pong':
                break;
        }
    }

    function updateConnectionStatus(connected) {
        const dot = document.querySelector('#status-connection .status-dot');
        const text = document.querySelector('#status-connection [data-i18n]');
        if (dot) {
            dot.classList.toggle('disconnected', !connected);
        }
        if (text) {
            const i18n = window.I18n || { t: k => k };
            text.textContent = connected
                ? i18n.t('status.connected')
                : i18n.t('status.disconnected');
        }
    }

    function updateProgress(data) {
        const fill = document.getElementById('progress-fill');
        const text = document.getElementById('progress-text');
        const container = document.getElementById('progress-container');

        if (container) container.classList.remove('hidden');

        const pct = Math.round((data.overall_progress || 0) * 100);
        if (fill) fill.style.width = pct + '%';
        if (text) text.textContent = `${pct}% — ${data.operation || ''}`;

        // Highlight executing node
        if (data.current_step !== undefined) {
            // Could highlight nodes here
        }
    }

    function onExecutionComplete(result) {
        const btn = document.getElementById('btn-execute');
        if (btn) btn.classList.remove('executing');

        const fill = document.getElementById('progress-fill');
        if (fill) fill.style.width = '100%';

        const i18n = window.I18n || { t: k => k };

        if (result && result.success) {
            const files = result.output_files || [];
            showToast(
                (i18n.t('toolbar.merge_complete') || 'Merge complete!') +
                (files.length ? ` (${files.length} files)` : ''),
                'success'
            );
        } else {
            showToast(
                (i18n.t('toolbar.merge_failed') || 'Merge failed: ') +
                (result.errors ? result.errors.join('; ') : 'Unknown error'),
                'error'
            );
        }

        // Hide progress after delay
        setTimeout(() => {
            const container = document.getElementById('progress-container');
            if (container) container.classList.add('hidden');
            if (fill) fill.style.width = '0%';
        }, 3000);
    }

    function onExecutionError(error) {
        const btn = document.getElementById('btn-execute');
        if (btn) btn.classList.remove('executing');

        showToast('Error: ' + error, 'error');

        setTimeout(() => {
            const container = document.getElementById('progress-container');
            if (container) container.classList.add('hidden');
        }, 3000);
    }

    // ─── Heartbeat ──────────────────────────────────────────────
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000);

    return {
        init,
        get,
        post,
    };
})();
