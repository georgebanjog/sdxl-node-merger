/* ═══════════════════════════════════════════════════════════════
   Connections — SVG Bezier curve connections between node ports
   ═══════════════════════════════════════════════════════════════ */

window.Connections = (() => {
    let connections = []; // { id, fromNode, fromOutput, toNode, toInput, type, element }
    let previewLine = null;
    let dragState = null; // { fromNode, fromOutput, fromPort, type }
    let idCounter = 0;

    function init() {
        // Listen for connection line clicks (to delete)
        document.addEventListener('mouseup', onGlobalMouseUp);
    }

    // ─── Connection Management ──────────────────────────────────
    function addConnection(fromNode, fromOutput, toNode, toInput, type) {
        // Check if already connected
        const existing = connections.find(c =>
            c.toNode === toNode && c.toInput === toInput
        );
        if (existing) removeConnection(existing.id);

        const id = 'conn_' + (++idCounter);
        const svg = Canvas.getSvgEl();

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const typeLower = type.toLowerCase();
        path.setAttribute('class', `connection-line conn-${type}`);
        path.setAttribute('data-id', id);
        path.style.stroke = `var(--conn-${typeLower})`;

        // Middle click to delete connection
        path.addEventListener('mousedown', (e) => {
            if (e.button === 1) {
                e.preventDefault();
                e.stopPropagation();
                removeConnection(id);
            }
        });

        svg.appendChild(path);

        const conn = { id, fromNode, fromOutput, toNode, toInput, type, element: path };
        connections.push(conn);

        updateConnection(conn);
        updatePortStates();
        updateStatusCounts();

        // Push to undo
        if (window.App && App.pushUndo) App.pushUndo();

        return conn;
    }

    function removeConnection(id) {
        const idx = connections.findIndex(c => c.id === id);
        if (idx === -1) return;

        const conn = connections[idx];
        conn.element.remove();
        connections.splice(idx, 1);

        updatePortStates();
        updateStatusCounts();

        if (window.App && App.pushUndo) App.pushUndo();
    }

    function removeConnectionsForNode(nodeId) {
        const toRemove = connections.filter(c => c.fromNode === nodeId || c.toNode === nodeId);
        for (const conn of toRemove) {
            conn.element.remove();
        }
        connections = connections.filter(c => c.fromNode !== nodeId && c.toNode !== nodeId);
        updatePortStates();
        updateStatusCounts();
    }

    function getConnectionsForNode(nodeId) {
        return connections.filter(c => c.fromNode === nodeId || c.toNode === nodeId);
    }

    // ─── Update Connection Path ─────────────────────────────────
    function updateConnection(conn) {
        const fromPort = getPortPosition(conn.fromNode, conn.fromOutput, 'output');
        const toPort = getPortPosition(conn.toNode, conn.toInput, 'input');

        if (!fromPort || !toPort) return;

        const d = buildBezierPath(fromPort.x, fromPort.y, toPort.x, toPort.y);
        conn.element.setAttribute('d', d);
    }

    function updateAll() {
        for (const conn of connections) {
            updateConnection(conn);
        }
        if (dragState && previewLine) {
            // Update preview line during drag
        }
    }

    function updateConnectionsForNode(nodeId) {
        const related = connections.filter(c => c.fromNode === nodeId || c.toNode === nodeId);
        for (const conn of related) {
            updateConnection(conn);
        }
    }

    // ─── Port Positions ─────────────────────────────────────────
    function getPortPosition(nodeId, portName, portType) {
        const node = window.Nodes ? Nodes.getNode(nodeId) : null;
        if (!node || !node.element) return null;

        const portEl = node.element.querySelector(
            `.node-port.${portType}-port[data-port="${portName}"]`
        );
        if (!portEl) return null;

        let offsetX = portEl.offsetWidth / 2;
        let offsetY = portEl.offsetHeight / 2;
        
        let current = portEl;
        while (current && current !== node.element) {
            offsetX += current.offsetLeft;
            offsetY += current.offsetTop;
            current = current.offsetParent;
        }

        return { x: node.x + offsetX, y: node.y + offsetY };
    }

    // ─── Bezier Path ────────────────────────────────────────────
    function buildBezierPath(x1, y1, x2, y2) {
        const dx = Math.abs(x2 - x1);
        const cpOffset = Math.max(50, dx * 0.4);

        return `M ${x1} ${y1} C ${x1 + cpOffset} ${y1}, ${x2 - cpOffset} ${y2}, ${x2} ${y2}`;
    }

    // ─── Drag to Connect ────────────────────────────────────────
    function startDrag(nodeId, portName, portType, portDataType, e) {
        const svg = Canvas.getSvgEl();

        previewLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const typeLower = portDataType.toLowerCase();
        previewLine.setAttribute('class', `connection-preview conn-${portDataType}`);
        previewLine.style.stroke = `var(--conn-${typeLower})`;
        svg.appendChild(previewLine);

        if (portType === 'output') {
            dragState = {
                fromNode: nodeId,
                fromOutput: portName,
                type: portDataType,
                direction: 'output',
            };
        } else {
            // Dragging from input — disconnect existing and reverse
            const existing = connections.find(c => c.toNode === nodeId && c.toInput === portName);
            if (existing) {
                dragState = {
                    fromNode: existing.fromNode,
                    fromOutput: existing.fromOutput,
                    type: existing.type,
                    direction: 'output',
                };
                removeConnection(existing.id);
            } else {
                dragState = {
                    toNode: nodeId,
                    toInput: portName,
                    type: portDataType,
                    direction: 'input',
                };
            }
        }

        document.addEventListener('mousemove', onDragMove);
        document.addEventListener('mouseup', onDragEnd);
    }

    function onDragMove(e) {
        if (!dragState || !previewLine) return;

        const canvasPos = Canvas.screenToCanvas(e.clientX, e.clientY);
        let fromPos, toPos;

        if (dragState.direction === 'output') {
            fromPos = getPortPosition(dragState.fromNode, dragState.fromOutput, 'output');
            toPos = canvasPos;
        } else {
            fromPos = canvasPos;
            toPos = getPortPosition(dragState.toNode, dragState.toInput, 'input');
        }

        if (fromPos && toPos) {
            const d = buildBezierPath(fromPos.x, fromPos.y, toPos.x, toPos.y);
            previewLine.setAttribute('d', d);
        }
    }

    function onDragEnd(e) {
        document.removeEventListener('mousemove', onDragMove);
        document.removeEventListener('mouseup', onDragEnd);

        if (previewLine) {
            previewLine.remove();
            previewLine = null;
        }

        if (!dragState) return;

        // Find port under cursor
        const el = document.elementFromPoint(e.clientX, e.clientY);
        const portEl = el ? el.closest('.node-port') : null;

        if (portEl) {
            const targetNodeEl = portEl.closest('.node');
            if (targetNodeEl) {
                const targetNodeId = targetNodeEl.dataset.nodeId;
                const targetPortName = portEl.dataset.port;
                const targetPortType = portEl.classList.contains('input-port') ? 'input' : 'output';
                const targetDataType = portEl.dataset.type;

                // Validate connection
                if (canConnect(dragState, targetNodeId, targetPortName, targetPortType, targetDataType)) {
                    if (dragState.direction === 'output' && targetPortType === 'input') {
                        addConnection(
                            dragState.fromNode, dragState.fromOutput,
                            targetNodeId, targetPortName,
                            dragState.type
                        );
                    } else if (dragState.direction === 'input' && targetPortType === 'output') {
                        addConnection(
                            targetNodeId, targetPortName,
                            dragState.toNode, dragState.toInput,
                            targetDataType
                        );
                    }
                }
            }
        }

        dragState = null;
    }

    function onGlobalMouseUp(e) {
        // Any middle-click on SVG connection lines handled via per-path listener
    }

    function canConnect(drag, targetNodeId, targetPortName, targetPortType, targetDataType) {
        // Can't connect to self
        if (drag.direction === 'output' && targetNodeId === drag.fromNode) return false;
        if (drag.direction === 'input' && targetNodeId === drag.toNode) return false;

        // Must connect output→input
        if (drag.direction === 'output' && targetPortType !== 'input') return false;
        if (drag.direction === 'input' && targetPortType !== 'output') return false;

        // Type must match
        if (drag.type !== targetDataType) return false;

        // No cycles (basic check)
        if (drag.direction === 'output') {
            if (wouldCreateCycle(drag.fromNode, targetNodeId)) return false;
        }

        return true;
    }

    function wouldCreateCycle(fromNode, toNode) {
        // BFS from toNode to see if we can reach fromNode
        const visited = new Set();
        const queue = [toNode];
        while (queue.length > 0) {
            const current = queue.shift();
            if (current === fromNode) return true;
            if (visited.has(current)) continue;
            visited.add(current);
            const outgoing = connections.filter(c => c.fromNode === current);
            for (const c of outgoing) queue.push(c.toNode);
        }
        return false;
    }

    // ─── Port State Updates ─────────────────────────────────────
    function updatePortStates() {
        // Reset all ports
        document.querySelectorAll('.node-port').forEach(p => p.classList.remove('connected'));

        // Mark connected ports
        for (const conn of connections) {
            const fromPort = document.querySelector(
                `.node[data-node-id="${conn.fromNode}"] .output-port[data-port="${conn.fromOutput}"]`
            );
            const toPort = document.querySelector(
                `.node[data-node-id="${conn.toNode}"] .input-port[data-port="${conn.toInput}"]`
            );
            if (fromPort) fromPort.classList.add('connected');
            if (toPort) toPort.classList.add('connected');
        }
    }

    function updateStatusCounts() {
        const el = document.getElementById('status-connections');
        if (el) {
            const label = el.querySelector('[data-i18n]');
            const text = label ? label.textContent : 'connections';
            el.innerHTML = `${connections.length} <span data-i18n="status.connections">${text}</span>`;
        }
    }

    // ─── Serialization ──────────────────────────────────────────
    function serialize() {
        return connections.map(c => ({
            from: { node: c.fromNode, output: c.fromOutput },
            to: { node: c.toNode, input: c.toInput },
            type: c.type,
        }));
    }

    function deserialize(data) {
        // Clear existing
        for (const c of [...connections]) {
            c.element.remove();
        }
        connections = [];
        idCounter = 0;

        for (const c of data) {
            addConnection(c.from.node, c.from.output, c.to.node, c.to.input, c.type || 'MODEL');
        }
    }

    function clear() {
        for (const c of [...connections]) {
            c.element.remove();
        }
        connections = [];
        updateStatusCounts();
    }

    return {
        init,
        addConnection,
        removeConnection,
        removeConnectionsForNode,
        getConnectionsForNode,
        updateAll,
        updateConnectionsForNode,
        updatePortStates,
        startDrag,
        serialize,
        deserialize,
        clear,
        getAll: () => [...connections],
    };
})();
