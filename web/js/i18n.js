/* ═══════════════════════════════════════════════════════════════
   I18n — Internationalization system
   ═══════════════════════════════════════════════════════════════ */

window.I18n = (() => {
    let currentLang = 'en';
    let translations = {};
    let fallback = {};

    async function init() {
        // Load config to get default language
        const config = await Api.get('/api/config');
        if (config && config.language) {
            currentLang = config.language;
        }

        // Load English as fallback
        fallback = await Api.get('/api/language?lang=en') || {};
        
        // Load current language
        if (currentLang !== 'en') {
            translations = await Api.get('/api/language?lang=' + currentLang) || {};
        } else {
            translations = fallback;
        }

        applyTranslations();
    }

    async function setLanguage(lang) {
        currentLang = lang;
        translations = await Api.get('/api/language?lang=' + lang) || {};
        applyTranslations();

        // Save to config
        Api.post('/api/config', { language: lang });
    }

    function t(key) {
        const keys = key.split('.');
        let val = translations;
        let fallbackVal = fallback;

        for (const k of keys) {
            val = val && val[k];
            fallbackVal = fallbackVal && fallbackVal[k];
        }

        return val || fallbackVal || key.split('.').pop().replace(/_/g, ' ');
    }

    function applyTranslations() {
        // Translate [data-i18n] elements
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            el.textContent = t(key);
        });

        // Translate [data-i18n-title] elements
        document.querySelectorAll('[data-i18n-title]').forEach(el => {
            const key = el.getAttribute('data-i18n-title');
            el.title = t(key);
        });

        // Translate [data-i18n-placeholder] elements
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            el.placeholder = t(key);
        });
    }

    return {
        init,
        setLanguage,
        t,
        applyTranslations,
        getCurrentLang: () => currentLang,
    };
})();
