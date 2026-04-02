/* ═══════════════════════════════════════════════════════════════
   Toolbar — Top toolbar actions
   ═══════════════════════════════════════════════════════════════ */

window.Toolbar = (() => {
    function init() {
        document.getElementById('btn-new-project').addEventListener('click', newProject);
        document.getElementById('btn-open-project').addEventListener('click', openProject);
        document.getElementById('btn-save-project').addEventListener('click', saveProject);
        document.getElementById('btn-undo').addEventListener('click', () => App.undo());
        document.getElementById('btn-redo').addEventListener('click', () => App.redo());
        document.getElementById('btn-execute').addEventListener('click', executeMerge);
        document.getElementById('btn-settings').addEventListener('click', openSettings);

        // Language selector
        loadLanguages();
        document.getElementById('select-language').addEventListener('change', (e) => {
            I18n.setLanguage(e.target.value);
        });

        // Theme selector
        loadThemes();
        document.getElementById('select-theme').addEventListener('change', (e) => {
            Themes.setTheme(e.target.value);
        });
    }

    async function loadLanguages() {
        const select = document.getElementById('select-language');
        const data = await Api.get('/api/languages');
        if (data && data.languages) {
            select.innerHTML = '';
            for (const lang of data.languages) {
                const opt = document.createElement('option');
                opt.value = lang.code;
                opt.textContent = lang.name;
                select.appendChild(opt);
            }
            // Set current
            const config = await Api.get('/api/config');
            if (config) select.value = config.language || 'en';
        }
    }

    async function loadThemes() {
        const select = document.getElementById('select-theme');
        const data = await Api.get('/api/themes');
        if (data && data.themes) {
            select.innerHTML = '';
            for (const theme of data.themes) {
                const opt = document.createElement('option');
                opt.value = theme.id;
                opt.textContent = theme.name;
                select.appendChild(opt);
            }
            const config = await Api.get('/api/config');
            if (config) select.value = config.theme || 'midnight';
        }
    }

    // ─── Project Actions ────────────────────────────────────────
    function newProject() {
        const i18n = window.I18n || { t: k => k };
        if (Nodes.getAllNodes().length > 0) {
            if (!confirm(i18n.t('toolbar.confirm_new'))) return;
        }
        Nodes.clear();
        Connections.clear();
        Canvas.setViewport({ x: 0, y: 0, zoom: 1.0 });
        App.setProjectName('');
        showToast(i18n.t('toolbar.new_created') || 'New project created', 'info');
    }

    async function openProject() {
        const i18n = window.I18n || { t: k => k };
        const data = await Api.get('/api/projects');
        if (!data || !data.projects) return;

        showModal(
            i18n.t('toolbar.open_project') || 'Open Project',
            () => {
                const body = document.getElementById('modal-body');
                body.innerHTML = '';

                if (data.projects.length === 0) {
                    body.innerHTML = `<p style="color: var(--text-tertiary); text-align: center; padding: 2rem;">${i18n.t('toolbar.no_projects') || 'No saved projects'}</p>`;
                    return;
                }

                const list = document.createElement('div');
                list.className = 'project-list';

                for (const proj of data.projects) {
                    const item = document.createElement('div');
                    item.className = 'project-item';
                    item.innerHTML = `
                        <div class="project-item-info">
                            <div class="project-item-name">${proj.name}</div>
                            <div class="project-item-meta">${proj.node_count} nodes · ${proj.modified || ''}</div>
                        </div>
                    `;

                    const delBtn = document.createElement('button');
                    delBtn.className = 'project-item-delete';
                    delBtn.textContent = '×';
                    delBtn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        await Api.post('/api/delete-project', { name: proj.name });
                        item.remove();
                    });
                    item.appendChild(delBtn);

                    item.addEventListener('click', async () => {
                        await Project.load(proj.name);
                        closeModal();
                    });

                    list.appendChild(item);
                }
                body.appendChild(list);
            }
        );
    }

    async function saveProject() {
        const i18n = window.I18n || { t: k => k };
        const currentName = App.getProjectName();

        showModal(
            i18n.t('toolbar.save_project') || 'Save Project',
            () => {
                const body = document.getElementById('modal-body');
                body.innerHTML = `
                    <div class="prop-group">
                        <div class="prop-group-title">${i18n.t('toolbar.project_name') || 'Project Name'}</div>
                        <input type="text" id="save-project-name" class="prop-input" value="${currentName}" placeholder="my_merge_project">
                    </div>
                `;
                const footer = document.getElementById('modal-footer');
                footer.innerHTML = '';
                const saveBtn = document.createElement('button');
                saveBtn.className = 'prop-btn prop-btn-primary';
                saveBtn.textContent = i18n.t('toolbar.save') || 'Save';
                saveBtn.addEventListener('click', async () => {
                    const name = document.getElementById('save-project-name').value.trim();
                    if (!name) return;
                    await Project.save(name);
                    App.setProjectName(name);
                    closeModal();
                    showToast(i18n.t('toolbar.saved') || 'Project saved!', 'success');
                });
                footer.appendChild(saveBtn);
            }
        );
    }

    // ─── Execute Merge ──────────────────────────────────────────
    async function executeMerge() {
        const i18n = window.I18n || { t: k => k };
        const btn = document.getElementById('btn-execute');

        // Validate
        const graph = {
            nodes: Nodes.serialize(),
            connections: Connections.serialize(),
        };

        const validation = await Api.post('/api/validate-graph', graph);
        if (validation && !validation.valid) {
            showToast(i18n.t('toolbar.validation_failed') || 'Validation failed: ' + validation.errors.join('; '), 'error');
            return;
        }

        btn.classList.add('executing');
        showProgress(true);

        try {
            await Api.post('/api/execute', graph);
            showToast(i18n.t('toolbar.execution_started') || 'Merge execution started!', 'info');
        } catch (e) {
            showToast('Error: ' + e.message, 'error');
            btn.classList.remove('executing');
            showProgress(false);
        }
    }

    // ─── Settings Modal ─────────────────────────────────────────
    async function openSettings() {
        const i18n = window.I18n || { t: k => k };
        const config = await Api.get('/api/config');
        if (!config) return;

        showModal(
            i18n.t('toolbar.settings') || 'Settings',
            () => {
                const body = document.getElementById('modal-body');
                body.innerHTML = '';

                // Directories
                const dirSection = document.createElement('div');
                dirSection.className = 'settings-section';
                dirSection.innerHTML = `<h4>📁 ${i18n.t('settings.directories') || 'Directories'}</h4>`;

                const dirs = [
                    { key: 'checkpoints', label: i18n.t('settings.checkpoints') || 'Checkpoints' },
                    { key: 'lora', label: i18n.t('settings.lora') || 'LoRA' },
                    { key: 'vae', label: i18n.t('settings.vae') || 'VAE' },
                    { key: 'output', label: i18n.t('settings.output') || 'Output' },
                ];

                for (const dir of dirs) {
                    const row = document.createElement('div');
                    row.className = 'settings-dir-row';
                    row.innerHTML = `<span class="settings-label">${dir.label}</span>`;
                    const inputRow = document.createElement('div');
                    inputRow.className = 'settings-dir-input-row';
                    const input = document.createElement('input');
                    input.type = 'text';
                    input.className = 'settings-dir-input';
                    input.id = 'dir-' + dir.key;
                    input.value = config.directories[dir.key] || '';
                    input.placeholder = i18n.t('settings.select_directory') || 'Select directory...';
                    inputRow.appendChild(input);

                    const browseBtn = document.createElement('button');
                    browseBtn.className = 'settings-browse-btn';
                    browseBtn.textContent = i18n.t('settings.browse') || 'Browse';
                    browseBtn.addEventListener('click', () => openDirBrowser(input));
                    inputRow.appendChild(browseBtn);

                    row.appendChild(inputRow);
                    dirSection.appendChild(row);
                }
                body.appendChild(dirSection);

                // Merge settings
                const mergeSection = document.createElement('div');
                mergeSection.className = 'settings-section';
                mergeSection.innerHTML = `<h4>⚙️ ${i18n.t('settings.merge') || 'Merge Settings'}</h4>`;

                // Device
                const deviceRow = document.createElement('div');
                deviceRow.className = 'settings-row';
                deviceRow.innerHTML = `<span class="settings-label">${i18n.t('settings.device') || 'Device'}</span>`;
                const deviceSelect = document.createElement('select');
                deviceSelect.className = 'prop-select';
                deviceSelect.id = 'setting-device';
                for (const d of ['auto', 'cpu', 'cuda']) {
                    const opt = document.createElement('option');
                    opt.value = d;
                    opt.textContent = d === 'auto' ? 'Auto (GPU if available)' : d.toUpperCase();
                    if (d === config.merge.device) opt.selected = true;
                    deviceSelect.appendChild(opt);
                }
                deviceRow.appendChild(deviceSelect);
                mergeSection.appendChild(deviceRow);

                // Default dtype
                const dtypeRow = document.createElement('div');
                dtypeRow.className = 'settings-row';
                dtypeRow.innerHTML = `<span class="settings-label">${i18n.t('settings.default_dtype') || 'Default Precision'}</span>`;
                const dtypeSelect = document.createElement('select');
                dtypeSelect.className = 'prop-select';
                dtypeSelect.id = 'setting-dtype';
                for (const dt of ['fp16', 'fp32', 'bf16']) {
                    const opt = document.createElement('option');
                    opt.value = dt;
                    opt.textContent = dt.toUpperCase();
                    if (dt === config.merge.default_dtype) opt.selected = true;
                    dtypeSelect.appendChild(opt);
                }
                dtypeRow.appendChild(dtypeSelect);
                mergeSection.appendChild(dtypeRow);

                // Low VRAM
                const lvRow = document.createElement('div');
                lvRow.className = 'settings-row';
                const lvLabel = document.createElement('label');
                lvLabel.className = 'prop-checkbox';
                const lvCb = document.createElement('input');
                lvCb.type = 'checkbox';
                lvCb.id = 'setting-low-vram';
                lvCb.checked = config.merge.low_vram;
                lvLabel.appendChild(lvCb);
                lvLabel.appendChild(document.createTextNode(' ' + (i18n.t('settings.low_vram') || 'Low VRAM Mode (slower, less memory)')));
                lvRow.appendChild(lvLabel);
                mergeSection.appendChild(lvRow);

                body.appendChild(mergeSection);

                // Save button
                const footer = document.getElementById('modal-footer');
                footer.innerHTML = '';
                const saveBtn = document.createElement('button');
                saveBtn.className = 'prop-btn prop-btn-primary';
                saveBtn.textContent = i18n.t('settings.save') || 'Save Settings';
                saveBtn.addEventListener('click', async () => {
                    const newConfig = {
                        directories: {
                            checkpoints: document.getElementById('dir-checkpoints').value,
                            lora: document.getElementById('dir-lora').value,
                            vae: document.getElementById('dir-vae').value,
                            output: document.getElementById('dir-output').value,
                        },
                        merge: {
                            device: document.getElementById('setting-device').value,
                            default_dtype: document.getElementById('setting-dtype').value,
                            low_vram: document.getElementById('setting-low-vram').checked,
                        },
                    };
                    await Api.post('/api/config', newConfig);
                    closeModal();
                    showToast(i18n.t('settings.saved') || 'Settings saved!', 'success');
                });
                footer.appendChild(saveBtn);
            }
        );
    }

    // ─── Directory Browser (in-modal) ───────────────────────────
    async function openDirBrowser(inputEl) {
        const i18n = window.I18n || { t: k => k };
        let currentPath = inputEl.value || '';

        const browser = document.createElement('div');
        browser.className = 'dir-browser';

        async function loadDir(path) {
            const data = await Api.post('/api/browse-directory', { path });
            if (!data) return;

            browser.innerHTML = '';

            const pathBar = document.createElement('div');
            pathBar.className = 'dir-browser-path';
            pathBar.textContent = data.current || '/';
            browser.appendChild(pathBar);

            const useBtn = document.createElement('button');
            useBtn.className = 'prop-btn prop-btn-primary';
            useBtn.textContent = i18n.t('settings.use_this') || 'Use this directory';
            useBtn.style.margin = '4px';
            useBtn.addEventListener('click', () => {
                inputEl.value = data.current;
                browser.remove();
            });
            browser.appendChild(useBtn);

            for (const item of (data.items || [])) {
                if (!item.is_dir) continue;
                const el = document.createElement('div');
                el.className = 'dir-item';
                el.innerHTML = `<span class="dir-item-icon">📁</span> ${item.name}`;
                el.addEventListener('click', () => loadDir(item.path));
                browser.appendChild(el);
            }
        }

        await loadDir(currentPath);
        inputEl.parentNode.insertBefore(browser, inputEl.nextSibling);
    }

    // ─── Modal Helpers ──────────────────────────────────────────
    function showModal(title, buildFn) {
        const overlay = document.getElementById('modal-overlay');
        document.getElementById('modal-title').textContent = title;
        document.getElementById('modal-body').innerHTML = '';
        document.getElementById('modal-footer').innerHTML = '';
        overlay.classList.remove('hidden');
        if (buildFn) buildFn();

        document.getElementById('modal-close').onclick = closeModal;
        overlay.onclick = (e) => {
            if (e.target === overlay) closeModal();
        };
    }

    function closeModal() {
        document.getElementById('modal-overlay').classList.add('hidden');
    }

    // ─── Toast ──────────────────────────────────────────────────
    function showToast(message, type = 'info') {
        window.showToast(message, type);
    }

    // ─── Progress ───────────────────────────────────────────────
    function showProgress(show) {
        const container = document.getElementById('progress-container');
        if (show) container.classList.remove('hidden');
        else container.classList.add('hidden');
    }

    return {
        init,
        closeModal,
        showModal,
        showProgress,
    };
})();

// ─── Global Toast Function ──────────────────────────────────────
window.showToast = function(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const msgSpan = document.createElement('span');
    msgSpan.className = 'toast-message';
    msgSpan.textContent = message;
    toast.appendChild(msgSpan);

    const closeBtn = document.createElement('button');
    closeBtn.className = 'toast-close';
    closeBtn.innerHTML = '&times;';
    closeBtn.addEventListener('click', () => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 200);
    });
    toast.appendChild(closeBtn);

    container.appendChild(toast);

    // Also log to browser console for debugging
    const prefix = type.toUpperCase();
    console.log(`[${prefix}] ${message}`);
};
