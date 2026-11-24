// Executes the actual deteksi.html client script against a lightweight DOM stub.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

let html = fs.readFileSync('templates/deteksi.html', 'utf8');
// Substitute Jinja placeholders with realistic defaults for offline VM testing
html = html.replace(/\{\{\s*model_info\.default_algorithm\s*\}\}/g, 'ensemble')
           .replace(/\{\{\s*model_info\.decision_threshold\s*\}\}/g, '0.5');
const source = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');

const elements = new Map();
function makeElement(id) {
    const classes = new Set();
    const children = [];
    return {
        id,
        textContent: '',
        innerHTML: '',
        value: '',
        src: '',
        title: '',
        disabled: false,
        dataset: {},
        style: {},
        handlers: {},
        children,
        classList: {
            add: (...names) => names.forEach(n => classes.add(n)),
            remove: (...names) => names.forEach(n => classes.delete(n)),
            toggle: (name, force) => (force === undefined ? (classes.has(name) ? classes.delete(name) : classes.add(name)) : (force ? classes.add(name) : classes.delete(name))),
            contains: name => classes.has(name),
        },
        set className(val) {
            classes.clear();
            String(val || '').split(/\s+/).filter(Boolean).forEach(n => classes.add(n));
        },
        get className() { return [...classes].join(' '); },
        appendChild(child) { children.push(child); return child; },
        append(...nodes) { children.push(...nodes); },
        addEventListener(type, cb) { this.handlers[type] = cb; },
        closest() { return null; },
    };
}

function getElement(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
}

let fetchResolve = null;
const context = vm.createContext({
    document: {
        getElementById: getElement,
        querySelectorAll: () => [],
        createElement: tag => makeElement(tag),
    },
    FormData: class { append() {} },
    Blob: class {},
    URL: { createObjectURL: () => 'blob:mock', revokeObjectURL: () => {} },
    navigator: { clipboard: { writeText: async () => {} } },
    setInterval: () => 1,
    clearInterval: () => {},
    setTimeout: cb => { cb(); return 1; },
    fetch: () => new Promise(resolve => { fetchResolve = resolve; }),
    console,
});

vm.runInContext(source, context);

// 1. Literal XSS-safe error rendering
context.showError('<img src=x onerror=alert(1)>');
assert.equal(getElement('validationAlert').textContent, '<img src=x onerror=alert(1)>');
assert.equal(getElement('validationAlert').innerHTML, '');

// 2. REAL verdict & exact proportional split bar alignment (60.1% Real vs 39.9% AI)
context.renderPredictionResult({
    label: 'UNCERTAIN',
    binary_verdict: 'REAL',
    decision_threshold: 0.5,
    prob_real: 0.601,
    prob_fake: 0.399,
    inference_ms: 42.3,
    algorithm: 'vit',
    algorithm_name: 'ViT-S/16',
    file_sha256: 'abcdef1234567890abcdef1234567890',
    forensic_metrics: { ela_mean: 2.1, noise_sigma: 8.4, fft_peak_z: 2.2 },
    key_findings: [{ category: 'Klasifikasi', status: 'ok', title: 'ViT-S/16', metric: '60.1%', detail: 'Non-AI' }],
    comparison: {
        vit: { id: 'vit', name: 'ViT-S/16', prob_real: 0.601, prob_fake: 0.399, latency_ms: 42.3, enabled: true },
        resnet18: { id: 'resnet18', name: 'ResNet-18', prob_real: 0.85, prob_fake: 0.15, latency_ms: 15.2, enabled: true },
    },
    visuals: { ela_map: 'data:image/jpeg;base64,AA==', fft_spectrum: 'data:image/jpeg;base64,BB==', gradcam_overlay: 'data:image/jpeg;base64,CC==' },
});
assert(getElement('verdictBanner').classList.contains('state-real'));
assert.equal(getElement('splitRealSeg').style.width, '60.10%');
assert.equal(getElement('splitFakeSeg').style.width, '39.90%');
assert.equal(getElement('probRealVal').textContent, '60,1%');
assert.equal(getElement('probFakeVal').textContent, '39,9%');
assert.equal(getElement('layerGradcam').disabled, false);
assert.equal(getElement('comparisonTableBody').children.length, 2);

// 3. C2PA AI_DISCLOSED verdict shows distinct C2PA banner & literal signer text
context.renderPredictionResult({
    label: 'AI_DISCLOSED',
    binary_verdict: 'AI_DISCLOSED',
    decision_threshold: 0.5,
    prob_real: 0.601,
    prob_fake: 0.399,
    inference_ms: 30,
    algorithm: 'vit',
    algorithm_name: 'ViT-S/16',
    summary: 'Penggunaan AI tercatat.',
    provenance: { status: 'trusted_ai', summary: 'C2PA valid', signer: '<script>xss</script>' },
    key_findings: [],
    comparison: {},
    visuals: {},
});
assert(getElement('verdictBanner').classList.contains('state-fake'));
assert(!getElement('c2paBanner').classList.contains('hidden'));
assert(getElement('c2paBannerText').textContent.includes('<script>xss</script>'));
assert.equal(getElement('c2paBannerText').innerHTML, '');

// 4. Stale-result cancellation on Reset mid-flight
(async () => {
    vm.runInContext('currentValidFile = {name: "sample.png", size: 1234}', context);
    const pending = context.executeAnalysis();
    context.resetAll();
    fetchResolve({ ok: true, json: async () => ({ label: 'FAKE', prob_real: 0.1, prob_fake: 0.9, inference_ms: 10 }) });
    await pending;
    assert(getElement('activeResultPanel').classList.contains('hidden'), 'Reset during scan must cancel stale result');
    console.log('Frontend tests passed: proportional split bar, C2PA disclosure, XSS safety, and stale cancellation.');
})().catch(err => { console.error(err); process.exitCode = 1; });
