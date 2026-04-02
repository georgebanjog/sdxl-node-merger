/* ═══════════════════════════════════════════════════════════════
   Sidebar — Node properties panel
   ═══════════════════════════════════════════════════════════════ */

window.Sidebar = (() => {
    let currentNode = null;

    function init() {
        const toggle = document.getElementById('sidebar-toggle');
        toggle.addEventListener('click', toggleSidebar);
    }

    function toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('sidebar-collapsed');
    }

    function showEmpty() {
        currentNode = null;
        document.getElementById('no-selection').classList.remove('hidden');
        document.getElementById('node-properties').classList.add('hidden');
        document.getElementById('node-properties').innerHTML = '';
    }

    function showNodeProperties(node) {
        if (!node) return showEmpty();
        currentNode = node;

        // Open sidebar if collapsed
        const sidebar = document.getElementById('sidebar');
        if (sidebar.classList.contains('sidebar-collapsed')) {
            sidebar.classList.remove('sidebar-collapsed');
        }

        document.getElementById('no-selection').classList.add('hidden');
        const props = document.getElementById('node-properties');
        props.classList.remove('hidden');
        props.innerHTML = '';

        const i18n = window.I18n || { t: k => k.split('.').pop() };

        // Title
        const titleGroup = createGroup(i18n.t('sidebar.node_info'));
        addInfoRow(titleGroup, i18n.t('sidebar.type'), i18n.t('nodes.' + node.type));
        addInfoRow(titleGroup, 'ID', node.id);
        props.appendChild(titleGroup);

        // Type-specific controls
        switch (node.type) {
            case 'checkpoint_loader':
            case 'vae_loader':
                buildFileSelector(props, node, node.type === 'vae_loader' ? 'vae' : 'checkpoints');
                break;
            case 'lora_loader':
                buildFileSelector(props, node, 'lora');
                buildStrengthSlider(props, node);
                break;
            case 'merge_models':
                buildMergeProperties(props, node);
                break;
            case 'apply_lora':
                buildStrengthSlider(props, node);
                break;
            case 'save_checkpoint':
                buildSaveProperties(props, node);
                break;
            case 'metadata_editor':
                buildMetadataEditor(props, node);
                break;
        }
    }

    // ─── Helper: Create property group ──────────────────────────
    function createGroup(title) {
        const group = document.createElement('div');
        group.className = 'prop-group';
        const titleEl = document.createElement('div');
        titleEl.className = 'prop-group-title';
        titleEl.textContent = title;
        group.appendChild(titleEl);
        return group;
    }

    function addInfoRow(group, label, value) {
        const row = document.createElement('div');
        row.className = 'prop-row';
        row.innerHTML = `<span class="prop-label">${label}</span><span class="prop-value">${value}</span>`;
        group.appendChild(row);
    }

    // ─── File Selector ──────────────────────────────────────────
    function buildFileSelector(container, node, dirType) {
        const i18n = window.I18n || { t: k => k };
        const group = createGroup(i18n.t('sidebar.file'));

        const select = document.createElement('select');
        select.className = 'prop-select';

        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = i18n.t('sidebar.select_file');
        select.appendChild(defaultOption);

        // Load files from server
        Api.get('/api/scan-models?type=' + dirType).then(data => {
            if (data && data.files) {
                for (const file of data.files) {
                    const opt = document.createElement('option');
                    opt.value = file.full_path;
                    opt.textContent = `${file.name} (${file.size})`;
                    if (file.full_path === node.data.file) opt.selected = true;
                    select.appendChild(opt);
                }
            }
            if (data && data.files && data.files.length === 0) {
                const opt = document.createElement('option');
                opt.disabled = true;
                opt.textContent = i18n.t('sidebar.no_files_found');
                select.appendChild(opt);
            }
        });

        select.addEventListener('change', () => {
            node.data.file = select.value;
            Nodes.updateNodeDisplay(node.id);
            if (window.App && App.pushUndo) App.pushUndo();
        });

        group.appendChild(select);

        // Show current path
        if (node.data.file) {
            const pathEl = document.createElement('div');
            pathEl.style.cssText = 'font-size: var(--font-size-xs); color: var(--text-tertiary); margin-top: 4px; word-break: break-all; font-family: var(--font-mono);';
            pathEl.textContent = node.data.file;
            group.appendChild(pathEl);
        }

        container.appendChild(group);
    }

    // ─── Strength Slider ────────────────────────────────────────
    function buildStrengthSlider(container, node) {
        const i18n = window.I18n || { t: k => k };
        const group = createGroup(i18n.t('sidebar.strength'));

        const row = document.createElement('div');
        row.className = 'prop-slider-row';

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.className = 'prop-slider';
        slider.min = '-2';
        slider.max = '2';
        slider.step = '0.01';
        slider.value = node.data.strength || 1.0;

        const valueDisplay = document.createElement('span');
        valueDisplay.className = 'prop-slider-value';
        valueDisplay.textContent = parseFloat(slider.value).toFixed(2);

        slider.addEventListener('input', () => {
            const val = parseFloat(slider.value);
            node.data.strength = val;
            valueDisplay.textContent = val.toFixed(2);
        });
        slider.addEventListener('change', () => {
            if (window.App && App.pushUndo) App.pushUndo();
        });

        row.appendChild(slider);
        row.appendChild(valueDisplay);
        group.appendChild(row);
        container.appendChild(group);
    }

    // ─── Merge Properties ───────────────────────────────────────
    function buildMergeProperties(container, node) {
        const i18n = window.I18n || { t: k => k };

        // Algorithm selector
        const algoGroup = createGroup(i18n.t('sidebar.algorithm'));
        const algos = Nodes.getAlgorithms();

        const algoSelect = document.createElement('select');
        algoSelect.className = 'prop-select';
        for (const algo of algos) {
            const opt = document.createElement('option');
            opt.value = algo.name;
            opt.textContent = algo.display_name;
            if (algo.name === node.data.algorithm) opt.selected = true;
            algoSelect.appendChild(opt);
        }
        algoSelect.addEventListener('change', () => {
            node.data.algorithm = algoSelect.value;
            // Reset params to defaults
            const algo = algos.find(a => a.name === algoSelect.value);
            if (algo) {
                node.data.params = {};
                for (const p of algo.params) {
                    node.data.params[p.name] = p.default;
                }
            }
            Nodes.updateNodeDisplay(node.id);
            showNodeProperties(node); // Rebuild sidebar
            if (window.App && App.pushUndo) App.pushUndo();
        });
        algoGroup.appendChild(algoSelect);

        // Algorithm description
        const algo = algos.find(a => a.name === node.data.algorithm);
        if (algo) {
            const desc = document.createElement('div');
            desc.style.cssText = 'font-size: var(--font-size-xs); color: var(--text-tertiary); margin-top: 4px; line-height: 1.4;';
            desc.textContent = algo.description;
            algoGroup.appendChild(desc);
        }
        container.appendChild(algoGroup);

        // Algorithm parameters
        if (algo && algo.params.length > 0) {
            const paramsGroup = createGroup(i18n.t('sidebar.parameters'));

            for (const param of algo.params) {
                const currentVal = node.data.params[param.name] !== undefined
                    ? node.data.params[param.name]
                    : param.default;

                if (param.type === 'float') {
                    const row = document.createElement('div');
                    row.innerHTML = `<span class="prop-label">${param.label}</span>`;
                    const sliderRow = document.createElement('div');
                    sliderRow.className = 'prop-slider-row';

                    const slider = document.createElement('input');
                    slider.type = 'range';
                    slider.className = 'prop-slider';
                    slider.min = String(param.min);
                    slider.max = String(param.max);
                    slider.step = String(param.step || 0.01);
                    slider.value = String(currentVal);

                    const display = document.createElement('span');
                    display.className = 'prop-slider-value';
                    display.textContent = parseFloat(currentVal).toFixed(2);

                    slider.addEventListener('input', () => {
                        node.data.params[param.name] = parseFloat(slider.value);
                        display.textContent = parseFloat(slider.value).toFixed(2);
                    });
                    slider.addEventListener('change', () => {
                        if (window.App && App.pushUndo) App.pushUndo();
                    });

                    sliderRow.appendChild(slider);
                    sliderRow.appendChild(display);
                    row.appendChild(sliderRow);
                    paramsGroup.appendChild(row);
                } else if (param.type === 'int') {
                    const row = document.createElement('div');
                    row.className = 'prop-row';
                    row.innerHTML = `<span class="prop-label">${param.label}</span>`;
                    const input = document.createElement('input');
                    input.type = 'number';
                    input.className = 'prop-input';
                    input.style.width = '100px';
                    input.min = String(param.min || 0);
                    input.max = String(param.max || 999999);
                    input.value = String(currentVal);
                    input.addEventListener('change', () => {
                        node.data.params[param.name] = parseInt(input.value);
                        if (window.App && App.pushUndo) App.pushUndo();
                    });
                    row.appendChild(input);
                    paramsGroup.appendChild(row);
                } else if (param.type === 'bool') {
                    const label = document.createElement('label');
                    label.className = 'prop-checkbox';
                    const cb = document.createElement('input');
                    cb.type = 'checkbox';
                    cb.checked = !!currentVal;
                    cb.addEventListener('change', () => {
                        node.data.params[param.name] = cb.checked;
                        if (window.App && App.pushUndo) App.pushUndo();
                    });
                    label.appendChild(cb);
                    label.appendChild(document.createTextNode(' ' + param.label));
                    paramsGroup.appendChild(label);
                } else if (param.type === 'select') {
                    const row = document.createElement('div');
                    row.innerHTML = `<span class="prop-label">${param.label}</span>`;
                    const sel = document.createElement('select');
                    sel.className = 'prop-select';
                    for (const opt of (param.options || [])) {
                        const o = document.createElement('option');
                        o.value = opt;
                        o.textContent = opt;
                        if (opt === currentVal) o.selected = true;
                        sel.appendChild(o);
                    }
                    sel.addEventListener('change', () => {
                        node.data.params[param.name] = sel.value;
                        if (window.App && App.pushUndo) App.pushUndo();
                    });
                    row.appendChild(sel);
                    paramsGroup.appendChild(row);
                }
            }
            container.appendChild(paramsGroup);
        }

        // MBW (Merge Block Weighted)
        const mbwGroup = createGroup(i18n.t('sidebar.mbw') || 'Block Weights (MBW)');
        const mbwCheck = document.createElement('label');
        mbwCheck.className = 'prop-checkbox';
        const mbwCb = document.createElement('input');
        mbwCb.type = 'checkbox';
        mbwCb.checked = !!node.data.use_mbw;
        mbwCb.addEventListener('change', () => {
            node.data.use_mbw = mbwCb.checked;
            mbwEditor.style.display = mbwCb.checked ? 'grid' : 'none';
            if (window.App && App.pushUndo) App.pushUndo();
        });
        mbwCheck.appendChild(mbwCb);
        mbwCheck.appendChild(document.createTextNode(' ' + (i18n.t('sidebar.enable_mbw') || 'Enable per-block weights')));
        mbwGroup.appendChild(mbwCheck);

        const mbwEditor = document.createElement('div');
        mbwEditor.className = 'mbw-editor';
        mbwEditor.style.display = node.data.use_mbw ? 'grid' : 'none';

        const mbwBlocks = [
            'IN00','IN01','IN02','IN03','IN04','IN05','IN06','IN07','IN08',
            'MID',
            'OUT00','OUT01','OUT02','OUT03','OUT04','OUT05','OUT06','OUT07','OUT08',
            'TE1','TE2','TIME_EMBED','LABEL_EMB','OUT_FINAL','VAE'
        ];
        const defaultAlpha = node.data.params.alpha || 0.5;

        for (const block of mbwBlocks) {
            const blockEl = document.createElement('div');
            blockEl.className = 'mbw-block';
            const label = document.createElement('div');
            label.className = 'mbw-block-label';
            label.textContent = block;
            const input = document.createElement('input');
            input.className = 'mbw-block-input';
            input.type = 'number';
            input.min = '0';
            input.max = '1';
            input.step = '0.01';
            input.value = node.data.mbw_weights[block] !== undefined ? node.data.mbw_weights[block] : defaultAlpha;
            input.addEventListener('change', () => {
                if (!node.data.mbw_weights) node.data.mbw_weights = {};
                node.data.mbw_weights[block] = parseFloat(input.value);
                if (window.App && App.pushUndo) App.pushUndo();
            });
            blockEl.appendChild(label);
            blockEl.appendChild(input);
            mbwEditor.appendChild(blockEl);
        }

        // "Set All" button
        const setAllRow = document.createElement('div');
        setAllRow.style.cssText = 'grid-column: 1 / -1; display: flex; gap: 4px; margin-top: 4px;';
        const setAllInput = document.createElement('input');
        setAllInput.className = 'mbw-block-input';
        setAllInput.type = 'number';
        setAllInput.min = '0';
        setAllInput.max = '1';
        setAllInput.step = '0.01';
        setAllInput.value = defaultAlpha;
        setAllInput.style.flex = '1';
        const setAllBtn = document.createElement('button');
        setAllBtn.className = 'prop-btn';
        setAllBtn.textContent = i18n.t('sidebar.set_all') || 'Set All';
        setAllBtn.style.width = 'auto';
        setAllBtn.addEventListener('click', () => {
            const val = parseFloat(setAllInput.value);
            mbwEditor.querySelectorAll('.mbw-block-input').forEach(inp => inp.value = val);
            for (const block of mbwBlocks) {
                node.data.mbw_weights[block] = val;
            }
        });
        setAllRow.appendChild(setAllInput);
        setAllRow.appendChild(setAllBtn);
        mbwEditor.appendChild(setAllRow);

        mbwGroup.appendChild(mbwEditor);
        container.appendChild(mbwGroup);
    }

    // ─── Save Properties ────────────────────────────────────────
    function buildSaveProperties(container, node) {
        const i18n = window.I18n || { t: k => k };

        // Filename
        const nameGroup = createGroup(i18n.t('sidebar.output'));
        const nameInput = document.createElement('input');
        nameInput.className = 'prop-input';
        nameInput.type = 'text';
        nameInput.value = node.data.filename || 'merged_model';
        nameInput.placeholder = 'merged_model';
        nameInput.addEventListener('change', () => {
            node.data.filename = nameInput.value;
            Nodes.updateNodeDisplay(node.id);
            if (window.App && App.pushUndo) App.pushUndo();
        });
        nameGroup.appendChild(nameInput);

        // Dtype
        const dtypeRow = document.createElement('div');
        dtypeRow.style.marginTop = '8px';
        dtypeRow.innerHTML = `<span class="prop-label">${i18n.t('sidebar.precision') || 'Precision'}</span>`;
        const dtypeSelect = document.createElement('select');
        dtypeSelect.className = 'prop-select';
        for (const dt of ['fp16', 'fp32', 'bf16']) {
            const opt = document.createElement('option');
            opt.value = dt;
            opt.textContent = dt.toUpperCase();
            if (dt === node.data.dtype) opt.selected = true;
            dtypeSelect.appendChild(opt);
        }
        dtypeSelect.addEventListener('change', () => {
            node.data.dtype = dtypeSelect.value;
            if (window.App && App.pushUndo) App.pushUndo();
        });
        dtypeRow.appendChild(dtypeSelect);
        nameGroup.appendChild(dtypeRow);

        container.appendChild(nameGroup);

        // Custom metadata
        buildMetadataEditor(container, node);
    }

    // ─── Metadata Editor ────────────────────────────────────────
    function buildMetadataEditor(container, node) {
        const i18n = window.I18n || { t: k => k };
        const group = createGroup(i18n.t('sidebar.metadata') || 'Metadata');

        if (!node.data.metadata) node.data.metadata = {};

        const table = document.createElement('table');
        table.className = 'metadata-table';
        table.innerHTML = '<thead><tr><th>Key</th><th>Value</th><th></th></tr></thead>';
        const tbody = document.createElement('tbody');

        function addMetaRow(key = '', value = '') {
            const tr = document.createElement('tr');
            const tdKey = document.createElement('td');
            const keyInput = document.createElement('input');
            keyInput.className = 'metadata-key-input';
            keyInput.value = key;
            keyInput.placeholder = 'key';
            tdKey.appendChild(keyInput);

            const tdVal = document.createElement('td');
            const valInput = document.createElement('input');
            valInput.className = 'metadata-value-input';
            valInput.value = value;
            valInput.placeholder = 'value';
            tdVal.appendChild(valInput);

            const tdDel = document.createElement('td');
            const delBtn = document.createElement('button');
            delBtn.className = 'project-item-delete';
            delBtn.textContent = '×';
            delBtn.addEventListener('click', () => {
                delete node.data.metadata[keyInput.value];
                tr.remove();
            });
            tdDel.appendChild(delBtn);

            // Update on change
            const updateMeta = () => {
                // Rebuild metadata from table
                node.data.metadata = {};
                tbody.querySelectorAll('tr').forEach(row => {
                    const k = row.querySelector('.metadata-key-input');
                    const v = row.querySelector('.metadata-value-input');
                    if (k && v && k.value) node.data.metadata[k.value] = v.value;
                });
            };
            keyInput.addEventListener('change', updateMeta);
            valInput.addEventListener('change', updateMeta);

            tr.appendChild(tdKey);
            tr.appendChild(tdVal);
            tr.appendChild(tdDel);
            tbody.appendChild(tr);
        }

        // Existing metadata
        for (const [k, v] of Object.entries(node.data.metadata)) {
            addMetaRow(k, v);
        }

        table.appendChild(tbody);
        group.appendChild(table);

        // Add button
        const addBtn = document.createElement('button');
        addBtn.className = 'prop-btn';
        addBtn.textContent = '+ ' + (i18n.t('sidebar.add_field') || 'Add Field');
        addBtn.style.marginTop = '4px';
        addBtn.addEventListener('click', () => addMetaRow());
        group.appendChild(addBtn);

        container.appendChild(group);
    }

    return {
        init,
        showEmpty,
        showNodeProperties,
        toggleSidebar,
    };
})();
