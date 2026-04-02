/* ═══════════════════════════════════════════════════════════════
   Themes — Color scheme management
   ═══════════════════════════════════════════════════════════════ */

window.Themes = (() => {
    let currentTheme = 'midnight';

    async function init() {
        const config = await Api.get('/api/config');
        if (config && config.theme) {
            currentTheme = config.theme;
            await applyTheme(currentTheme);
        }
    }

    async function setTheme(name) {
        currentTheme = name;
        await applyTheme(name);
        Api.post('/api/config', { theme: name });
    }

    async function applyTheme(name) {
        const data = await Api.get('/api/theme?name=' + name);
        if (!data || !data.colors) return;

        const root = document.documentElement;
        for (const [key, value] of Object.entries(data.colors)) {
            root.style.setProperty('--' + key, value);
        }
    }

    return {
        init,
        setTheme,
        getCurrentTheme: () => currentTheme,
    };
})();
