/* ═══════════════════════════════════════════════════════════════
   App — Main application controller
   ═══════════════════════════════════════════════════════════════ */

window.App = (() => {
    let projectName = '';
    let undoStack = [];
    let redoStack = [];
    const MAX_UNDO = 50;
    let undoTimer = null;

    async function init() {
        console.log('SDXL Node Merger — Initializing...');

        // Init modules in order
        Api.init();
        await I18n.init();
        await Themes.init();
        Canvas.init();
        Connections.init();
        Nodes.init();
        Sidebar.init();
        Toolbar.init();

        // Load algorithms
        const algoData = await Api.get('/api/algorithms');
        if (algoData && algoData.algorithms) {
            Nodes.setAlgorithms(algoData.algorithms);
        }

        // Check device info
        const config = await Api.get('/api/config');
        if (config) {
            const deviceEl = document.getElementById('status-device');
            if (deviceEl) {
                const device = config.merge.device === 'auto' ? 'Auto' : config.merge.device.toUpperCase();
                const lowVram = config.merge.low_vram ? ' (Low VRAM)' : '';
                deviceEl.textContent = `${device}${lowVram}`;
            }
        }

        // Apply initial translations
        I18n.applyTranslations();

        // Push initial empty state
        pushUndo();

        console.log('SDXL Node Merger — Ready!');
    }

    // ─── Undo / Redo ────────────────────────────────────────────
    function pushUndo() {
        // Debounce to avoid rapid-fire saves
        if (undoTimer) clearTimeout(undoTimer);
        undoTimer = setTimeout(() => {
            const state = getState();
            undoStack.push(state);
            if (undoStack.length > MAX_UNDO) undoStack.shift();
            redoStack = [];
        }, 100);
    }

    function undo() {
        if (undoStack.length <= 1) return;
        const current = undoStack.pop();
        redoStack.push(current);
        const prev = undoStack[undoStack.length - 1];
        restoreState(prev);
    }

    function redo() {
        if (redoStack.length === 0) return;
        const state = redoStack.pop();
        undoStack.push(state);
        restoreState(state);
    }

    function getState() {
        return JSON.stringify({
            nodes: Nodes.serialize(),
            connections: Connections.serialize(),
        });
    }

    function restoreState(stateStr) {
        try {
            const state = JSON.parse(stateStr);
            // Temporarily disable undo pushing
            const origPush = window.App.pushUndo;
            window.App.pushUndo = () => {};

            Nodes.clear();
            Connections.clear();
            if (state.nodes) Nodes.deserialize(state.nodes);
            setTimeout(() => {
                if (state.connections) Connections.deserialize(state.connections);
                window.App.pushUndo = origPush;
            }, 30);
        } catch (e) {
            console.error('Failed to restore state:', e);
        }
    }

    // ─── Project Name ───────────────────────────────────────────
    function setProjectName(name) {
        projectName = name;
        document.title = name ? `${name} — SDXL Node Merger` : 'SDXL Node Merger';
    }

    function getProjectName() {
        return projectName;
    }

    // ─── Init on DOM Ready ──────────────────────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    return {
        init,
        pushUndo,
        undo,
        redo,
        setProjectName,
        getProjectName,
    };
})();
