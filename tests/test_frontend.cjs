// No browser dependencies: execute the actual page script with a small DOM stub.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('templates/deteksi.html', 'utf8');
const source = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
const elements = new Map();
function element(id) {
    if (!elements.has(id)) {
        const classes = new Set();
        elements.set(id, {textContent: '', innerHTML: '', style: {}, handlers: {},
            classList: {add: (...names) => names.forEach(n => classes.add(n)),
                        remove: (...names) => names.forEach(n => classes.delete(n)),
                        contains: name => classes.has(name)},
            addEventListener(type, callback) { this.handlers[type] = callback; }});
    }
    return elements.get(id);
}
const timers = [];
let fetchCalls = 0;
const context = vm.createContext({document: {getElementById: element},
    setTimeout: callback => timers.push(callback), performance: {now: () => 0},
    window: {location: {protocol: 'http:'}}, fetch: () => { fetchCalls++; },
    console});
vm.runInContext(source, context);
for (const [label, style] of [['UNCERTAIN', 'result-uncertain'], ['REAL', 'result-real'], ['FAKE', 'result-fake']]) {
    context.renderPredictionResult({label, prob_real: 0.5, prob_fake: 0.5,
        inference_ms: 10, summary: 'Evidence-limited result', warnings: ['Partial edits unvalidated']});
    assert(element('activeResultPanel').classList.contains(style));
    assert.equal(element('diagDetailText').textContent, 'Partial edits unvalidated');
}
assert(!element('activeResultPanel').classList.contains('result-uncertain'));
context.showError('<img src=x onerror=alert(1)>');
assert.equal(element('validationAlert').textContent, '<img src=x onerror=alert(1)>');
assert.equal(element('validationAlert').innerHTML, '');
context.renderPredictionResult({label: 'UNCERTAIN', prob_real: 0.6, prob_fake: 0.4,
    inference_ms: 20, architecture: 'Community Forensics ViT-S/16',
    score_type: 'uncalibrated_sigmoid', summary: 'Uncertain', warnings: []});
assert.equal(element('diagTexture').textContent, 'Community Forensics ViT-S/16');
assert.equal(element('diagFrequency').textContent, 'Sigmoid belum dikalibrasi');
context.renderPredictionResult({label: 'AI_DISCLOSED', prob_real: 0.9999, prob_fake: 0.0001,
    inference_ms: 20, architecture: 'Community Forensics ViT-S/16',
    score_type: 'uncalibrated_sigmoid', summary: 'AI disclosed by credentials', warnings: [],
    provenance: {summary: 'Trusted AI disclosure', signer: '<img src=x onerror=alert(1)>'}});
assert.equal(element('resultMainTitle').textContent, 'Penggunaan AI tercatat');
assert.equal(element('confidenceNumber').textContent, '0,0%');
assert.equal(element('confidenceTag').textContent, 'SKOR VISUAL SAJA');
assert(element('provenanceSummary').textContent.includes('<img src=x onerror=alert(1)>'));
assert.equal(element('provenanceSummary').innerHTML, '');
assert(element('activeResultPanel').classList.contains('result-fake'));

(async () => {
    vm.runInContext('currentValidFile = {name: "old.png"}', context);
    const analysis = element('btnAnalyze').handlers.click();
    context.resetAll();
    timers.shift()();
    await analysis;
    assert.equal(fetchCalls, 0, 'Reset during animation must cancel stale analysis');
    assert(element('activeResultPanel').classList.contains('hidden'));
    console.log('Frontend: three verdict states, literal errors, and stale-result cancellation passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
