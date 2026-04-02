/* ═══════════════════════════════════════════════════════════════
   Project — Save/Load project state
   ═══════════════════════════════════════════════════════════════ */

window.Project = (() => {
    async function save(name) {
        const project = {
            version: '1.0',
            name: name,
            created: new Date().toISOString(),
            modified: new Date().toISOString(),
            viewport: Canvas.getViewport(),
            nodes: Nodes.serialize(),
            connections: Connections.serialize(),
        };

        await Api.post('/api/save-project', { name, project });
        return project;
    }

    async function load(name) {
        const data = await Api.get('/api/load-project?name=' + encodeURIComponent(name));
        if (!data) {
            showToast('Failed to load project', 'error');
            return false;
        }

        // Clear current state
        Nodes.clear();
        Connections.clear();

        // Restore viewport
        if (data.viewport) {
            Canvas.setViewport(data.viewport);
        }

        // Restore nodes
        if (data.nodes) {
            Nodes.deserialize(data.nodes);
        }

        // Restore connections (after small delay to ensure DOM is ready)
        setTimeout(() => {
            if (data.connections) {
                Connections.deserialize(data.connections);
            }
            Canvas.updateTransform();
        }, 50);

        App.setProjectName(data.name || name);
        showToast(`Project "${name}" loaded`, 'success');
        return true;
    }

    function exportToJSON() {
        const project = {
            version: '1.0',
            name: App.getProjectName() || 'export',
            created: new Date().toISOString(),
            modified: new Date().toISOString(),
            viewport: Canvas.getViewport(),
            nodes: Nodes.serialize(),
            connections: Connections.serialize(),
        };

        const blob = new Blob([JSON.stringify(project, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = (project.name || 'project') + '.json';
        a.click();
        URL.revokeObjectURL(url);
    }

    function importFromJSON(file) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    Nodes.clear();
                    Connections.clear();
                    if (data.viewport) Canvas.setViewport(data.viewport);
                    if (data.nodes) Nodes.deserialize(data.nodes);
                    setTimeout(() => {
                        if (data.connections) Connections.deserialize(data.connections);
                    }, 50);
                    App.setProjectName(data.name || '');
                    showToast('Project imported', 'success');
                    resolve(true);
                } catch (err) {
                    showToast('Invalid project file', 'error');
                    resolve(false);
                }
            };
            reader.readAsText(file);
        });
    }

    return { save, load, exportToJSON, importFromJSON };
})();
