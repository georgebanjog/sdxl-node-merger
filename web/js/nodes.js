/* ═══════════════════════════════════════════════════════════════
   Nodes — Node creation, management, selection, and interaction
   ═══════════════════════════════════════════════════════════════ */

window.Nodes = (() => {
    let nodes = {}; // id -> node object
    let idCounter = 0;
    let selectedNodes = new Set();
    let dragState = null;
    let algorithmsList = [];

    // ─── Node Type Icons (SVG) ──────────────────────────────────
    const NODE_ICONS = {
        checkpoint_loader: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="4"/><path d="M8 12h8M12 8v8"/></svg>',
        lora_loader: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v12M6 12h12"/></svg>',
        vae_loader: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>',
        merge_models: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>',
        apply_lora: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20V10M18 20V4M6 20v-4"/></svg>',
        replace_vae: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>',
        save_checkpoint: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/></svg>',
        metadata_editor: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>',
        note: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    };

    // ─── Node Type Definitions ──────────────────────────────────
    const NODE_DEFS = {
        checkpoint_loader: {
            title: 'nodes.checkpoint_loader',
            category: 'source',
            inputs: [],
            outputs: [{ name: 'MODEL', type: 'MODEL' }],
            defaults: { file: '' },
        },
        lora_loader: {
            title: 'nodes.lora_loader',
            category: 'source',
            inputs: [],
            outputs: [{ name: 'LORA', type: 'LORA' }],
            defaults: { file: '', strength: 1.0 },
        },
        vae_loader: {
            title: 'nodes.vae_loader',
            category: 'source',
            inputs: [],
            outputs: [{ name: 'VAE', type: 'VAE' }],
            defaults: { file: '' },
        },
        merge_models: {
            title: 'nodes.merge_models',
            category: 'processing',
            inputs: [
                { name: 'MODEL_A', type: 'MODEL' },
                { name: 'MODEL_B', type: 'MODEL' },
            ],
            outputs: [{ name: 'MODEL', type: 'MODEL' }],
            defaults: { algorithm: 'weighted_sum', params: { alpha: 0.5 }, use_mbw: false, mbw_weights: {} },
            dynamicInputs: true,
        },
        apply_lora: {
            title: 'nodes.apply_lora',
            category: 'processing',
            inputs: [
                { name: 'MODEL', type: 'MODEL' },
                { name: 'LORA', type: 'LORA' },
            ],
            outputs: [{ name: 'MODEL', type: 'MODEL' }],
            defaults: { strength: 1.0 },
        },
        replace_vae: {
            title: 'nodes.replace_vae',
            category: 'processing',
            inputs: [
                { name: 'MODEL', type: 'MODEL' },
                { name: 'VAE', type: 'VAE' },
            ],
            outputs: [{ name: 'MODEL', type: 'MODEL' }],
            defaults: {},
        },
        save_checkpoint: {
            title: 'nodes.save_checkpoint',
            category: 'output',
            inputs: [{ name: 'MODEL', type: 'MODEL' }],
            outputs: [],
            defaults: { filename: 'merged_model', dtype: 'fp16', metadata: {} },
        },
        metadata_editor: {
            title: 'nodes.metadata_editor',
            category: 'utility',
            inputs: [{ name: 'MODEL', type: 'MODEL' }],
            outputs: [{ name: 'MODEL', type: 'MODEL' }],
            defaults: { metadata: {} },
        },
        note: {
            title: 'nodes.note',
            category: 'utility',
            inputs: [],
            outputs: [],
            defaults: { text: '' },
        },
    };

    // ─── Init ───────────────────────────────────────────────────
    function init() {
        // Load algorithms list from server
        Api.get('/api/algorithms').then(data => {
            if (data && data.algorithms) {
                algorithmsList = data.algorithms;
            }
        });

        // Global key listeners
        document.addEventListener('keydown', onKeyDown);
    }

    function setAlgorithms(algos) {
        algorithmsList = algos;
    }

    // ─── Node Creation ──────────────────────────────────────────
    function createNode(type, x, y, data = null, id = null) {
        const def = NODE_DEFS[type];
        if (!def) {
            console.error('Unknown node type:', type);
            return null;
        }

        const nodeId = id || ('node_' + (++idCounter));
        // Ensure idCounter stays ahead
        if (id) {
            const num = parseInt(id.replace('node_', ''));
            if (!isNaN(num) && num >= idCounter) idCounter = num + 1;
        }

        const nodeData = data || { ...def.defaults };

        // Check if merge needs MODEL_C
        let inputs = [...def.inputs];
        if (type === 'merge_models' && nodeData.algorithm) {
            const algo = algorithmsList.find(a => a.name === nodeData.algorithm);
            if (algo && algo.num_models === 3) {
                inputs = [
                    { name: 'MODEL_A', type: 'MODEL' },
                    { name: 'MODEL_B', type: 'MODEL' },
                    { name: 'MODEL_C', type: 'MODEL' },
                ];
            }
        }

        const node = {
            id: nodeId,
            type,
            x: x || 0,
            y: y || 0,
            data: nodeData,
            inputs: inputs,
            outputs: def.outputs,
            element: null,
        };

        // Create DOM element
        node.element = buildNodeElement(node);
        Canvas.getViewportEl().appendChild(node.element);

        nodes[nodeId] = node;
        updateNodePosition(node);
        updateStatusCounts();

        if (window.App && App.pushUndo) App.pushUndo();

        return node;
    }

    function buildNodeElement(node) {
        const def = NODE_DEFS[node.type];
        const i18n = window.I18n || { t: (k) => k.split('.').pop() };

        const el = document.createElement('div');
        el.className = `node node-type-${node.type} node-category-${def.category}`;
        el.dataset.nodeId = node.id;

        // Header
        const header = document.createElement('div');
        header.className = 'node-header';
        header.innerHTML = `
            <span class="node-header-icon">${NODE_ICONS[node.type] || ''}</span>
            <span class="node-header-title">${i18n.t(def.title)}</span>
        `;
        el.appendChild(header);

        // Body
        const body = document.createElement('div');
        body.className = 'node-body';

        // Input ports
        for (const input of node.inputs) {
            const row = document.createElement('div');
            row.className = 'node-port-row input-row';
            row.innerHTML = `
                <span class="node-port input-port port-${input.type}" data-port="${input.name}" data-type="${input.type}"></span>
                <span class="node-port-label">${input.name}</span>
            `;
            body.appendChild(row);
        }

        // Inline controls based on type
        buildInlineControls(body, node);

        // Output ports
        for (const output of node.outputs) {
            const row = document.createElement('div');
            row.className = 'node-port-row output-row';
            row.innerHTML = `
                <span class="node-port-label">${output.name}</span>
                <span class="node-port output-port port-${output.type}" data-port="${output.name}" data-type="${output.type}"></span>
            `;
            body.appendChild(row);
        }

        el.appendChild(body);

        // ─── Event Handlers ─────────────────────────────────────
        // Drag node (left click on header)
        header.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            e.stopPropagation();
            startDragNode(node, e);
        });

        // Select node (left click)
        el.addEventListener('mousedown', (e) => {
            if (e.button === 0) {
                e.stopPropagation();
                if (e.shiftKey) {
                    toggleSelect(node.id);
                } else if (!selectedNodes.has(node.id)) {
                    selectNode(node.id);
                }
            }
            // Middle click = delete
            if (e.button === 1) {
                e.preventDefault();
                e.stopPropagation();
                deleteNode(node.id);
            }
        });

        // Port interactions
        el.querySelectorAll('.node-port').forEach(port => {
            port.addEventListener('mousedown', (e) => {
                if (e.button !== 0) return;
                e.stopPropagation();
                const portName = port.dataset.port;
                const portType = port.classList.contains('input-port') ? 'input' : 'output';
                const dataType = port.dataset.type;
                Connections.startDrag(node.id, portName, portType, dataType, e);
            });
        });

        // Context menu
        el.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();
            showNodeContextMenu(node, e.clientX, e.clientY);
        });

        return el;
    }

    function buildInlineControls(body, node) {
        const i18n = window.I18n || { t: (k) => k };

        if (node.type === 'checkpoint_loader' || node.type === 'vae_loader') {
            const ctrl = document.createElement('div');
            ctrl.className = 'node-control';
            const display = document.createElement('div');
            display.className = `node-file-display ${node.data.file ? '' : 'no-file'}`;
            display.textContent = node.data.file ? node.data.file.split(/[/\\]/).pop() : i18n.t('nodes.select_file');
            display.onclick = () => selectNode(node.id);
            ctrl.appendChild(display);
            body.appendChild(ctrl);
        }

        if (node.type === 'lora_loader') {
            const ctrl = document.createElement('div');
            ctrl.className = 'node-control';
            const display = document.createElement('div');
            display.className = `node-file-display ${node.data.file ? '' : 'no-file'}`;
            display.textContent = node.data.file ? node.data.file.split(/[/\\]/).pop() : i18n.t('nodes.select_file');
            display.onclick = () => selectNode(node.id);
            ctrl.appendChild(display);
            body.appendChild(ctrl);
        }

        if (node.type === 'merge_models') {
            const ctrl = document.createElement('div');
            ctrl.className = 'node-control';
            const algoDisplay = document.createElement('div');
            algoDisplay.className = 'node-file-display';
            const algo = algorithmsList.find(a => a.name === node.data.algorithm);
            algoDisplay.textContent = algo ? algo.display_name : node.data.algorithm;
            algoDisplay.onclick = () => selectNode(node.id);
            ctrl.appendChild(algoDisplay);
            body.appendChild(ctrl);
        }

        if (node.type === 'save_checkpoint') {
            const ctrl = document.createElement('div');
            ctrl.className = 'node-control';
            const nameDisplay = document.createElement('div');
            nameDisplay.className = 'node-file-display';
            nameDisplay.textContent = node.data.filename || 'merged_model';
            nameDisplay.onclick = () => selectNode(node.id);
            ctrl.appendChild(nameDisplay);
            body.appendChild(ctrl);
        }

        if (node.type === 'note') {
            const textarea = document.createElement('textarea');
            textarea.className = 'node-note-text';
            textarea.value = node.data.text || '';
            textarea.placeholder = i18n.t('nodes.note_placeholder') || 'Write a note...';
            textarea.addEventListener('input', () => {
                node.data.text = textarea.value;
            });
            textarea.addEventListener('mousedown', e => e.stopPropagation());
            body.appendChild(textarea);
        }
    }

    // ─── Node Position ──────────────────────────────────────────
    function updateNodePosition(node) {
        if (!node.element) return;
        node.element.style.left = node.x + 'px';
        node.element.style.top = node.y + 'px';
    }

    // ─── Node Dragging ──────────────────────────────────────────
    function startDragNode(node, e) {
        // If node not selected, select it first
        if (!selectedNodes.has(node.id)) {
            deselectAll();
            selectNode(node.id);
        }

        const startX = e.clientX;
        const startY = e.clientY;
        const zoom = Canvas.getViewport().zoom;

        // Store initial positions of all selected nodes
        const initialPositions = {};
        for (const nid of selectedNodes) {
            const n = nodes[nid];
            if (n) initialPositions[nid] = { x: n.x, y: n.y };
        }

        node.element.classList.add('dragging');

        const onMove = (ev) => {
            const dx = (ev.clientX - startX) / zoom;
            const dy = (ev.clientY - startY) / zoom;

            for (const nid of selectedNodes) {
                const n = nodes[nid];
                if (n && initialPositions[nid]) {
                    n.x = initialPositions[nid].x + dx;
                    n.y = initialPositions[nid].y + dy;
                    updateNodePosition(n);
                    Connections.updateConnectionsForNode(nid);
                }
            }
        };

        const onUp = () => {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            node.element.classList.remove('dragging');
            if (window.App && App.pushUndo) App.pushUndo();
        };

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    }

    // ─── Selection ──────────────────────────────────────────────
    function selectNode(nodeId) {
        deselectAll();
        selectedNodes.add(nodeId);
        const node = nodes[nodeId];
        if (node && node.element) {
            node.element.classList.add('selected');
        }
        if (window.Sidebar) Sidebar.showNodeProperties(node);
    }

    function toggleSelect(nodeId) {
        if (selectedNodes.has(nodeId)) {
            selectedNodes.delete(nodeId);
            const n = nodes[nodeId];
            if (n && n.element) n.element.classList.remove('selected');
        } else {
            selectedNodes.add(nodeId);
            const n = nodes[nodeId];
            if (n && n.element) n.element.classList.add('selected');
        }
    }

    function deselectAll() {
        for (const nid of selectedNodes) {
            const n = nodes[nid];
            if (n && n.element) n.element.classList.remove('selected');
        }
        selectedNodes.clear();
    }

    function selectInRect(screenX, screenY, width, height) {
        for (const node of Object.values(nodes)) {
            const screenPos = Canvas.canvasToScreen(node.x, node.y);
            const nodeW = node.element ? node.element.offsetWidth * Canvas.getViewport().zoom : 200;
            const nodeH = node.element ? node.element.offsetHeight * Canvas.getViewport().zoom : 100;

            const nodeRight = screenPos.x + nodeW;
            const nodeBottom = screenPos.y + nodeH;
            const selRight = screenX + width;
            const selBottom = screenY + height;

            if (screenPos.x < selRight && nodeRight > screenX &&
                screenPos.y < selBottom && nodeBottom > screenY) {
                selectedNodes.add(node.id);
                node.element.classList.add('selected');
            }
        }
    }

    // ─── Delete ─────────────────────────────────────────────────
    function deleteNode(nodeId) {
        const node = nodes[nodeId];
        if (!node) return;

        Connections.removeConnectionsForNode(nodeId);
        if (node.element) node.element.remove();
        delete nodes[nodeId];
        selectedNodes.delete(nodeId);
        updateStatusCounts();

        if (window.Sidebar) Sidebar.showEmpty();
        if (window.App && App.pushUndo) App.pushUndo();
    }

    function deleteSelected() {
        const toDelete = [...selectedNodes];
        for (const nid of toDelete) {
            deleteNode(nid);
        }
    }

    // ─── Update Node (when properties change) ───────────────────
    function updateNodeDisplay(nodeId) {
        const node = nodes[nodeId];
        if (!node || !node.element) return;

        // Re-build the element
        const parent = node.element.parentNode;
        const wasSelected = selectedNodes.has(nodeId);
        node.element.remove();

        // Check if merge algorithm changed and needs MODEL_C
        if (node.type === 'merge_models') {
            const algo = algorithmsList.find(a => a.name === node.data.algorithm);
            if (algo && algo.num_models === 3) {
                node.inputs = [
                    { name: 'MODEL_A', type: 'MODEL' },
                    { name: 'MODEL_B', type: 'MODEL' },
                    { name: 'MODEL_C', type: 'MODEL' },
                ];
            } else {
                node.inputs = [
                    { name: 'MODEL_A', type: 'MODEL' },
                    { name: 'MODEL_B', type: 'MODEL' },
                ];
            }
        }

        node.element = buildNodeElement(node);
        if (wasSelected) node.element.classList.add('selected');
        parent.appendChild(node.element);
        updateNodePosition(node);
        Connections.updateConnectionsForNode(nodeId);
    }

    // ─── Context Menu ───────────────────────────────────────────
    function showNodeContextMenu(node, x, y) {
        const i18n = window.I18n || { t: k => k };
        const menu = document.getElementById('context-menu');
        const items = document.getElementById('context-menu-items');

        items.innerHTML = '';

        const menuItems = [
            { label: i18n.t('context.duplicate'), action: () => duplicateNode(node.id) },
            { label: i18n.t('context.select_connected'), action: () => selectConnected(node.id) },
            { separator: true },
            { label: i18n.t('context.delete'), action: () => deleteNode(node.id), class: 'danger', shortcut: 'Mid Click' },
        ];

        for (const item of menuItems) {
            if (item.separator) {
                const sep = document.createElement('div');
                sep.className = 'context-menu-separator';
                items.appendChild(sep);
                continue;
            }

            const el = document.createElement('div');
            el.className = 'context-menu-item' + (item.class ? ' ' + item.class : '');
            el.innerHTML = `
                <span>${item.label}</span>
                ${item.shortcut ? `<span class="context-menu-shortcut">${item.shortcut}</span>` : ''}
            `;
            el.addEventListener('click', () => {
                menu.classList.add('hidden');
                item.action();
            });
            items.appendChild(el);
        }

        menu.classList.remove('hidden');
        menu.style.left = x + 'px';
        menu.style.top = y + 'px';

        // Close on click outside
        const close = (ev) => {
            if (!menu.contains(ev.target)) {
                menu.classList.add('hidden');
                document.removeEventListener('mousedown', close);
            }
        };
        setTimeout(() => document.addEventListener('mousedown', close), 10);
    }

    function duplicateNode(nodeId) {
        const node = nodes[nodeId];
        if (!node) return;
        createNode(node.type, node.x + 30, node.y + 30, { ...node.data });
    }

    function selectConnected(nodeId) {
        const conns = Connections.getConnectionsForNode(nodeId);
        for (const c of conns) {
            const otherId = c.fromNode === nodeId ? c.toNode : c.fromNode;
            selectedNodes.add(otherId);
            const n = nodes[otherId];
            if (n && n.element) n.element.classList.add('selected');
        }
    }

    // ─── Keyboard Shortcuts ─────────────────────────────────────
    function onKeyDown(e) {
        // Don't handle if typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

        if (e.key === 'Delete' || e.key === 'Backspace') {
            deleteSelected();
        }
        if (e.key === 'a' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            for (const node of Object.values(nodes)) {
                selectedNodes.add(node.id);
                node.element.classList.add('selected');
            }
        }
        if (e.key === 'c' && (e.ctrlKey || e.metaKey)) {
            copySelected();
        }
        if (e.key === 'v' && (e.ctrlKey || e.metaKey)) {
            pasteNodes();
        }
        if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
            if (window.App) App.undo();
        }
        if ((e.key === 'z' && (e.ctrlKey || e.metaKey) && e.shiftKey) ||
            (e.key === 'y' && (e.ctrlKey || e.metaKey))) {
            if (window.App) App.redo();
        }
        if (e.key === 'f' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            Canvas.fitToNodes();
        }
    }

    // ─── Copy / Paste ───────────────────────────────────────────
    let clipboard = null;

    function copySelected() {
        if (selectedNodes.size === 0) return;
        const nodesCopy = [];
        for (const nid of selectedNodes) {
            const n = nodes[nid];
            if (n) {
                nodesCopy.push({
                    type: n.type,
                    x: n.x,
                    y: n.y,
                    data: JSON.parse(JSON.stringify(n.data)),
                });
            }
        }
        clipboard = nodesCopy;
    }

    function pasteNodes() {
        if (!clipboard) return;
        deselectAll();
        for (const nc of clipboard) {
            const node = createNode(nc.type, nc.x + 40, nc.y + 40, JSON.parse(JSON.stringify(nc.data)));
            if (node) {
                selectedNodes.add(node.id);
                node.element.classList.add('selected');
            }
        }
    }

    // ─── Status ─────────────────────────────────────────────────
    function updateStatusCounts() {
        const el = document.getElementById('status-nodes');
        if (el) {
            const label = el.querySelector('[data-i18n]');
            const text = label ? label.textContent : 'nodes';
            el.innerHTML = `${Object.keys(nodes).length} <span data-i18n="status.nodes">${text}</span>`;
        }
    }

    // ─── Serialization ──────────────────────────────────────────
    function serialize() {
        return Object.values(nodes).map(n => ({
            id: n.id,
            type: n.type,
            position: { x: n.x, y: n.y },
            data: n.data,
        }));
    }

    function deserialize(data) {
        clear();
        for (const nd of data) {
            createNode(nd.type, nd.position.x, nd.position.y, nd.data, nd.id);
        }
    }

    function clear() {
        for (const node of Object.values(nodes)) {
            if (node.element) node.element.remove();
        }
        nodes = {};
        selectedNodes.clear();
        idCounter = 0;
        updateStatusCounts();
    }

    return {
        init,
        createNode,
        deleteNode,
        deleteSelected,
        selectNode,
        deselectAll,
        toggleSelect,
        selectInRect,
        getNode: (id) => nodes[id],
        getAllNodes: () => Object.values(nodes),
        getSelectedIds: () => [...selectedNodes],
        updateNodeDisplay,
        setAlgorithms,
        serialize,
        deserialize,
        clear,
        getAlgorithms: () => algorithmsList,
    };
})();
