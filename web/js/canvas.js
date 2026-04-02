/* ═══════════════════════════════════════════════════════════════
   Canvas Controller — Zoom, Pan, Grid, and Interaction Management
   ═══════════════════════════════════════════════════════════════ */

window.Canvas = (() => {
    // ─── State ──────────────────────────────────────────────────
    let viewport = { x: 0, y: 0, zoom: 1.0 };
    const ZOOM_MIN = 0.15;
    const ZOOM_MAX = 3.0;
    const ZOOM_STEP = 0.08;

    let isPanning = false;
    let panStart = { x: 0, y: 0 };
    let isSelecting = false;
    let selectStart = { x: 0, y: 0 };
    let selectionBox = null;

    // DOM elements
    let wrapper, viewportEl, bgEl, svgEl;

    // ─── Init ───────────────────────────────────────────────────
    function init() {
        wrapper = document.getElementById('canvas-wrapper');
        viewportEl = document.getElementById('canvas-viewport');
        bgEl = document.getElementById('canvas-bg');
        svgEl = document.getElementById('connections-svg');

        wrapper.addEventListener('wheel', onWheel, { passive: false });
        wrapper.addEventListener('mousedown', onMouseDown);
        wrapper.addEventListener('mousemove', onMouseMove);
        wrapper.addEventListener('mouseup', onMouseUp);
        wrapper.addEventListener('dblclick', onDoubleClick);
        wrapper.addEventListener('contextmenu', onContextMenu);

        // Prevent default context menu
        wrapper.addEventListener('contextmenu', e => e.preventDefault());

        updateTransform();
    }

    // ─── Transform ──────────────────────────────────────────────
    function updateTransform() {
        const tx = `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`;
        viewportEl.style.transform = tx;

        // Update grid background
        const gridSize = 20 * viewport.zoom;
        const gridMajor = 100 * viewport.zoom;
        const ox = viewport.x % gridSize;
        const oy = viewport.y % gridSize;
        const omx = viewport.x % gridMajor;
        const omy = viewport.y % gridMajor;

        bgEl.style.backgroundImage = `
            radial-gradient(circle at ${50 + viewport.x * 0.01}% ${50 + viewport.y * 0.01}%, rgba(59,130,246,0.03) 0%, transparent 70%),
            linear-gradient(var(--canvas-grid-major) 1px, transparent 1px),
            linear-gradient(90deg, var(--canvas-grid-major) 1px, transparent 1px),
            linear-gradient(var(--canvas-grid) 1px, transparent 1px),
            linear-gradient(90deg, var(--canvas-grid) 1px, transparent 1px)
        `;
        bgEl.style.backgroundSize = `100% 100%, ${gridMajor}px ${gridMajor}px, ${gridMajor}px ${gridMajor}px, ${gridSize}px ${gridSize}px, ${gridSize}px ${gridSize}px`;
        bgEl.style.backgroundPosition = `0 0, ${omx}px ${omy}px, ${omx}px ${omy}px, ${ox}px ${oy}px, ${ox}px ${oy}px`;

        // SVG is inside viewport, inherits transform automatically

        // Update zoom display
        const zoomEl = document.getElementById('zoom-level');
        if (zoomEl) zoomEl.textContent = Math.round(viewport.zoom * 100) + '%';
    }

    // ─── Screen ↔ Canvas Coordinates ────────────────────────────
    function screenToCanvas(sx, sy) {
        const rect = wrapper.getBoundingClientRect();
        return {
            x: (sx - rect.left - viewport.x) / viewport.zoom,
            y: (sy - rect.top - viewport.y) / viewport.zoom,
        };
    }

    function canvasToScreen(cx, cy) {
        const rect = wrapper.getBoundingClientRect();
        return {
            x: cx * viewport.zoom + viewport.x + rect.left,
            y: cy * viewport.zoom + viewport.y + rect.top,
        };
    }

    // ─── Event Handlers ─────────────────────────────────────────
    function onWheel(e) {
        e.preventDefault();

        const rect = wrapper.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
        const newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, viewport.zoom + delta * viewport.zoom));

        // Zoom toward mouse position
        const ratio = newZoom / viewport.zoom;
        viewport.x = mx - (mx - viewport.x) * ratio;
        viewport.y = my - (my - viewport.y) * ratio;
        viewport.zoom = newZoom;

        updateTransform();
        if (window.Connections) Connections.updateAll();
    }

    function onMouseDown(e) {
        // Right mouse button = pan
        if (e.button === 2) {
            e.preventDefault();
            isPanning = true;
            panStart.x = e.clientX - viewport.x;
            panStart.y = e.clientY - viewport.y;
            wrapper.style.cursor = 'grabbing';
            return;
        }

        // Left click on empty canvas = deselect or start selection
        if (e.button === 0 && e.target === wrapper || e.target === bgEl || e.target === viewportEl) {
            if (!e.shiftKey) {
                if (window.Nodes) Nodes.deselectAll();
                if (window.Sidebar) Sidebar.showEmpty();
            }

            // Start selection box
            isSelecting = true;
            const pos = screenToCanvas(e.clientX, e.clientY);
            selectStart.x = e.clientX;
            selectStart.y = e.clientY;

            selectionBox = document.createElement('div');
            selectionBox.className = 'selection-box';
            selectionBox.style.left = e.clientX - wrapper.getBoundingClientRect().left + 'px';
            selectionBox.style.top = e.clientY - wrapper.getBoundingClientRect().top + 'px';
            selectionBox.style.width = '0px';
            selectionBox.style.height = '0px';
            wrapper.appendChild(selectionBox);
        }

        // Middle click on a node = delete it (handled in Nodes.js)
    }

    function onMouseMove(e) {
        if (isPanning) {
            viewport.x = e.clientX - panStart.x;
            viewport.y = e.clientY - panStart.y;
            updateTransform();
            if (window.Connections) Connections.updateAll();
            return;
        }

        if (isSelecting && selectionBox) {
            const rect = wrapper.getBoundingClientRect();
            const x = Math.min(e.clientX, selectStart.x) - rect.left;
            const y = Math.min(e.clientY, selectStart.y) - rect.top;
            const w = Math.abs(e.clientX - selectStart.x);
            const h = Math.abs(e.clientY - selectStart.y);

            selectionBox.style.left = x + 'px';
            selectionBox.style.top = y + 'px';
            selectionBox.style.width = w + 'px';
            selectionBox.style.height = h + 'px';
        }
    }

    function onMouseUp(e) {
        if (isPanning && e.button === 2) {
            isPanning = false;
            wrapper.style.cursor = '';
            return;
        }

        if (isSelecting && selectionBox) {
            // Select nodes within box
            const rect = wrapper.getBoundingClientRect();
            const box = {
                left: parseInt(selectionBox.style.left),
                top: parseInt(selectionBox.style.top),
                width: parseInt(selectionBox.style.width),
                height: parseInt(selectionBox.style.height),
            };

            if (box.width > 5 && box.height > 5 && window.Nodes) {
                Nodes.selectInRect(
                    box.left + rect.left,
                    box.top + rect.top,
                    box.width,
                    box.height
                );
            }

            selectionBox.remove();
            selectionBox = null;
            isSelecting = false;
        }
    }

    function onDoubleClick(e) {
        if (e.target === wrapper || e.target === bgEl || e.target === viewportEl) {
            // Show quick-add menu
            showQuickAddMenu(e.clientX, e.clientY);
        }
    }

    function onContextMenu(e) {
        e.preventDefault();
    }

    // ─── Quick Add Menu ─────────────────────────────────────────
    function showQuickAddMenu(clientX, clientY) {
        const menu = document.getElementById('quick-add-menu');
        const rect = wrapper.getBoundingClientRect();

        menu.classList.remove('hidden');
        menu.style.left = (clientX - rect.left) + 'px';
        menu.style.top = (clientY - rect.top) + 'px';

        const searchInput = document.getElementById('quick-add-search');
        searchInput.value = '';
        searchInput.focus();

        populateQuickAddList('');

        // Close on click outside
        const closeHandler = (ev) => {
            if (!menu.contains(ev.target)) {
                menu.classList.add('hidden');
                document.removeEventListener('mousedown', closeHandler);
            }
        };
        setTimeout(() => document.addEventListener('mousedown', closeHandler), 10);

        // Store position for node creation
        menu._createPos = screenToCanvas(clientX, clientY);

        // Filter on typing
        searchInput.oninput = () => populateQuickAddList(searchInput.value);

        // Handle keyboard
        searchInput.onkeydown = (ev) => {
            if (ev.key === 'Escape') {
                menu.classList.add('hidden');
            } else if (ev.key === 'Enter') {
                const first = menu.querySelector('.quick-add-item');
                if (first) first.click();
            }
        };
    }

    function populateQuickAddList(filter) {
        const list = document.getElementById('quick-add-list');
        const menu = document.getElementById('quick-add-menu');
        list.innerHTML = '';

        const i18n = window.I18n || { t: (k) => k };

        const nodeTypes = [
            { type: 'checkpoint_loader', category: 'source', color: 'var(--node-checkpoint)', label: i18n.t('nodes.checkpoint_loader') },
            { type: 'lora_loader', category: 'source', color: 'var(--node-lora)', label: i18n.t('nodes.lora_loader') },
            { type: 'vae_loader', category: 'source', color: 'var(--node-vae)', label: i18n.t('nodes.vae_loader') },
            { type: 'merge_models', category: 'processing', color: 'var(--node-merge)', label: i18n.t('nodes.merge_models') },
            { type: 'apply_lora', category: 'processing', color: 'var(--node-lora)', label: i18n.t('nodes.apply_lora') },
            { type: 'replace_vae', category: 'processing', color: 'var(--node-vae)', label: i18n.t('nodes.replace_vae') },
            { type: 'save_checkpoint', category: 'output', color: 'var(--node-output)', label: i18n.t('nodes.save_checkpoint') },
            { type: 'metadata_editor', category: 'utility', color: 'var(--node-utility)', label: i18n.t('nodes.metadata_editor') },
            { type: 'note', category: 'utility', color: 'var(--node-utility)', label: i18n.t('nodes.note') },
        ];

        const filterLower = filter.toLowerCase();
        const categories = {};

        for (const nt of nodeTypes) {
            if (filter && !nt.label.toLowerCase().includes(filterLower) && !nt.type.includes(filterLower)) {
                continue;
            }
            if (!categories[nt.category]) categories[nt.category] = [];
            categories[nt.category].push(nt);
        }

        for (const [cat, items] of Object.entries(categories)) {
            const catEl = document.createElement('div');
            catEl.className = 'quick-add-category';
            catEl.textContent = i18n.t('categories.' + cat) || cat;
            list.appendChild(catEl);

            for (const item of items) {
                const itemEl = document.createElement('div');
                itemEl.className = 'quick-add-item';
                itemEl.innerHTML = `
                    <span class="node-color-dot" style="background: ${item.color}"></span>
                    <span class="node-label">${item.label}</span>
                `;
                itemEl.addEventListener('click', () => {
                    const pos = menu._createPos;
                    if (window.Nodes) Nodes.createNode(item.type, pos.x, pos.y);
                    menu.classList.add('hidden');
                });
                list.appendChild(itemEl);
            }
        }
    }

    // ─── Public API ─────────────────────────────────────────────
    return {
        init,
        screenToCanvas,
        canvasToScreen,
        getViewport: () => ({ ...viewport }),
        setViewport: (v) => { viewport = { ...viewport, ...v }; updateTransform(); },
        getWrapperEl: () => wrapper,
        getViewportEl: () => viewportEl,
        getSvgEl: () => svgEl,
        updateTransform,
        fitToNodes: () => {
            if (!window.Nodes) return;
            const nodes = Nodes.getAllNodes();
            if (nodes.length === 0) return;
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            for (const n of nodes) {
                minX = Math.min(minX, n.x);
                minY = Math.min(minY, n.y);
                maxX = Math.max(maxX, n.x + 220);
                maxY = Math.max(maxY, n.y + 120);
            }
            const rect = wrapper.getBoundingClientRect();
            const padding = 100;
            const contentW = maxX - minX + padding * 2;
            const contentH = maxY - minY + padding * 2;
            const zoom = Math.min(rect.width / contentW, rect.height / contentH, 1.5);
            viewport.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom));
            viewport.x = rect.width / 2 - (minX + maxX) / 2 * viewport.zoom;
            viewport.y = rect.height / 2 - (minY + maxY) / 2 * viewport.zoom;
            updateTransform();
            if (window.Connections) Connections.updateAll();
        },
    };
})();
