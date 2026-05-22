/**
 * NEXUS AI Recommender — Master Application Script
 * Handles persona tab switching, dataset fetching, recommendations,
 * code copying, live stats, dataset upload, and AI/ML experiments.
 */

document.addEventListener('DOMContentLoaded', () => {

    // ========================================================
    //  PARAMETER HELP TOOLTIPS
    // ========================================================
    const tooltip = document.createElement('div');
    tooltip.className = 'param-tooltip';
    document.body.appendChild(tooltip);

    let tipTimeout;
    document.addEventListener('mouseover', e => {
        const btn = e.target.closest('.param-help');
        if (!btn) return;
        clearTimeout(tipTimeout);
        tooltip.textContent = btn.dataset.tip || '';
        const r = btn.getBoundingClientRect();
        let left = r.left + r.width / 2 - 140;
        let top  = r.bottom + 8 + window.scrollY;
        left = Math.max(8, Math.min(left, window.innerWidth - 296));
        tooltip.style.left = left + 'px';
        tooltip.style.top  = top + 'px';
        tipTimeout = setTimeout(() => tooltip.classList.add('visible'), 20);
    });
    document.addEventListener('mouseout', e => {
        const btn = e.target.closest('.param-help');
        if (!btn) return;
        clearTimeout(tipTimeout);
        tooltip.classList.remove('visible');
    });

    // ========================================================
    //  PERSONA TAB SWITCHING
    // ========================================================
    const personaBtns = document.querySelectorAll('.persona-btn');
    const views = document.querySelectorAll('.view');

    personaBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.target;
            personaBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            views.forEach(v => {
                v.classList.remove('active');
                v.classList.add('hidden');
            });
            const targetView = document.getElementById(target);
            if (targetView) {
                targetView.classList.add('active');
                targetView.classList.remove('hidden');
            }
            if (target === 'view-aiml') loadLiveStats();
        });
    });

    // ========================================================
    //  DATASET TABLE — fetch and render
    // ========================================================
    async function loadDataset() {
        try {
            const dsSelect = document.getElementById('dataset-selector');
            const dsValue = dsSelect ? dsSelect.value : 'tech';

            const downloadBtn = document.getElementById('download-dataset-btn');
            if (downloadBtn) downloadBtn.href = `/dataset/synthetic/download?dataset=${dsValue}`;

            const resp = await fetch(`/dataset/sample?dataset=${dsValue}`);
            if (!resp.ok) throw new Error(`Status ${resp.status}`);
            const data = await resp.json();

            const thead = document.getElementById('dataset-thead');
            const tbody = document.getElementById('dataset-tbody');

            thead.innerHTML = '<tr>' + data.columns.map(c =>
                `<th>${c.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</th>`
            ).join('') + '</tr>';

            const colClasses = {
                user_id:      'col-user_id',
                user_name:    'col-user_name',
                product_id:   'col-product_id',
                product_name: 'col-product_name',
                category:     'col-category',
                rating:       'col-rating',
                date:         'col-date',
                sentiment:    'col-sentiment',
                item_id:      'col-item_id',
                implicit:     'col-implicit',
                timestamp:    'col-timestamp',
                user_idx:     'col-user_idx',
                item_idx:     'col-item_idx',
            };

            tbody.innerHTML = data.rows.map(row =>
                '<tr>' + data.columns.map(c => {
                    let val = row[c];
                    let extra = '';
                    if (c === 'rating') {
                        const stars = '★'.repeat(Number(val)) + '☆'.repeat(5 - Number(val));
                        val = `${val} ${stars}`;
                    }
                    if (c === 'sentiment') extra = ` data-val="${val}"`;
                    return `<td class="${colClasses[c] || ''}"${extra}>${val}</td>`;
                }).join('') + '</tr>'
            ).join('');

            document.getElementById('meta-rows').textContent = `${data.total_sampled} rows loaded`;
            document.getElementById('table-row-count').textContent =
                `Showing all ${data.total_sampled} rows · Click any User ID to auto-fill the search`;

        } catch (err) {
            document.getElementById('dataset-tbody').innerHTML =
                `<tr><td colspan="8" class="loading-cell" style="color:#f43f5e">⚠️ Could not load dataset: ${err.message}</td></tr>`;
        }
    }

    loadDataset();

    const datasetSelector = document.getElementById('dataset-selector');
    if (datasetSelector) {
        datasetSelector.addEventListener('change', loadDataset);
    }

    // ========================================================
    //  TOAST
    // ========================================================
    const toast = document.getElementById('toast');
    function showToast(msg) {
        toast.textContent = msg;
        toast.classList.remove('hidden');
        clearTimeout(toast._timer);
        toast._timer = setTimeout(() => toast.classList.add('hidden'), 3000);
    }

    // ========================================================
    //  END SHOPPER — Recommendations
    // ========================================================
    const recommendBtn   = document.getElementById('recommend-btn');
    const refreshBtn     = document.getElementById('refresh-btn');
    const userIdInput    = document.getElementById('user-id-input');
    const loading        = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const productGrid    = document.getElementById('product-grid');
    const modelBadge     = document.getElementById('model-badge');
    const resultsSubText = document.getElementById('results-sub-text');

    let currentVariant = null;
    let currentUserId  = null;

    async function fetchRecommendations() {
        const userId = userIdInput.value.trim();
        if (!userId) { showToast('Please enter a User ID'); return; }

        currentUserId = userId;
        loading.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        recommendBtn.disabled = true;
        if (refreshBtn) refreshBtn.disabled = true;

        try {
            const resp = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: userId, n_recommendations: 8, filter_purchased: true })
            });
            if (!resp.ok) throw new Error(`API Error ${resp.status}`);
            const data = await resp.json();

            currentVariant = data.variant;

            if (data.variant === 'v1') {
                modelBadge.textContent = '⚡ Model V1 — SVD Baseline';
                modelBadge.className = 'model-badge v1';
                if (resultsSubText) resultsSubText.textContent = 'Served by Collaborative Filtering (ALS/SVD)';
            } else {
                modelBadge.textContent = '🧬 Model V2 — Hybrid AI';
                modelBadge.className = 'model-badge v2';
                if (resultsSubText) resultsSubText.textContent = 'Served by Hybrid (ALS + LightFM) Recommender';
            }

            const CAT_META = {
                'Audio':           { icon: '🎧', color: '#a855f7,#6366f1' },
                'Gaming':          { icon: '🎮', color: '#f43f5e,#e11d48' },
                'Electronics':     { icon: '⚡', color: '#3b82f6,#6366f1' },
                'Furniture':       { icon: '🛋️', color: '#f59e0b,#ef4444' },
                'Office Supplies': { icon: '📋', color: '#10b981,#059669' },
                'Technology':      { icon: '💻', color: '#06b6d4,#3b82f6' },
                'Decor':           { icon: '🪴', color: '#84cc16,#10b981' },
                'Appliances':      { icon: '☕', color: '#f97316,#ef4444' },
                'Tablets':         { icon: '📱', color: '#8b5cf6,#6366f1' },
                'Wearables':       { icon: '⌚', color: '#ec4899,#a855f7' },
                'Peripherals':     { icon: '🖱️', color: '#14b8a6,#06b6d4' },
                'Displays':        { icon: '🖥️', color: '#64748b,#475569' },
            };

            productGrid.innerHTML = '';
            data.recommendations.forEach((item, index) => {
                const card = document.createElement('div');
                card.className = 'product-card product-card-text';
                card.style.setProperty('--card-index', index);
                card.addEventListener('click', () => logClick(item.item_id));

                const pct = Math.min(item.score * 100, 100).toFixed(1);
                const displayName = item.product_name || item.item_id;
                const category = item.category || 'Electronics';
                const meta = CAT_META[category] || { icon: '🏷️', color: '#6366f1,#8b5cf6' };
                const [c1, c2] = meta.color.split(',');

                card.innerHTML = `
                    <div class="pcard-accent" style="background: linear-gradient(135deg, ${c1}, ${c2})">
                        <span class="pcard-icon">${meta.icon}</span>
                        <span class="pcard-rank">#${item.rank}</span>
                    </div>
                    <div class="pcard-body">
                        <div class="pcard-category">${category}</div>
                        <div class="pcard-name">${displayName}</div>
                        <div class="pcard-score-row">
                            <span class="pcard-pct" style="color:${c1}">${pct}%</span>
                            <span class="pcard-label">AI Match</span>
                        </div>
                        <div class="pcard-bar">
                            <div class="pcard-fill" style="width:${pct}%; background: linear-gradient(90deg, ${c1}, ${c2})"></div>
                        </div>
                    </div>`;
                productGrid.appendChild(card);
            });

            loading.classList.add('hidden');
            resultsSection.classList.remove('hidden');

        } catch (err) {
            showToast(`Error: ${err.message}`);
            loading.classList.add('hidden');
        } finally {
            recommendBtn.disabled = false;
            if (refreshBtn) refreshBtn.disabled = false;
        }
    }

    recommendBtn.addEventListener('click', fetchRecommendations);
    if (refreshBtn) refreshBtn.addEventListener('click', fetchRecommendations);
    userIdInput.addEventListener('keydown', e => { if (e.key === 'Enter') fetchRecommendations(); });

    document.getElementById('dataset-tbody').addEventListener('click', e => {
        const td = e.target.closest('td.col-user_id');
        if (td) {
            const uid = td.textContent.trim();
            userIdInput.value = uid;
            showToast(`User ID "${uid}" selected — click Get Recommendations!`);
        }
    });

    // ========================================================
    //  A/B CLICK LOGGER
    // ========================================================
    async function logClick(itemId) {
        if (!currentVariant || !currentUserId) return;
        showToast(`Logged click for ${currentVariant.toUpperCase()} — feeding A/B engine!`);
        try {
            await fetch('/ab-test/click', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: currentUserId, item_id: itemId, variant: currentVariant })
            });
        } catch (e) { console.error('Click log failed', e); }
    }

    // ========================================================
    //  A/B STATS MODAL
    // ========================================================
    const statsBtn   = document.getElementById('stats-btn');
    const statsModal = document.getElementById('stats-modal');
    const closeBtn   = document.querySelector('.close-btn');

    async function loadABStats() {
        try {
            const resp = await fetch('/ab-test/stats');
            const stats = await resp.json();

            document.getElementById('v1-impressions').textContent = stats.v1_impressions;
            document.getElementById('v1-clicks').textContent      = stats.v1_clicks;
            document.getElementById('v1-ctr').textContent         = (stats.v1_ctr * 100).toFixed(2) + '%';
            document.getElementById('v2-impressions').textContent = stats.v2_impressions;
            document.getElementById('v2-clicks').textContent      = stats.v2_clicks;
            document.getElementById('v2-ctr').textContent         = (stats.v2_ctr * 100).toFixed(2) + '%';
            document.getElementById('current-winner').textContent = stats.winner || 'Need more data';

            const maxCtr = Math.max(stats.v1_ctr, stats.v2_ctr, 0.01);
            document.getElementById('v1-ctr-bar').style.width = `${(stats.v1_ctr / maxCtr) * 100}%`;
            document.getElementById('v2-ctr-bar').style.width = `${(stats.v2_ctr / maxCtr) * 100}%`;

        } catch (err) {
            document.getElementById('current-winner').textContent = 'Error';
        }
    }

    statsBtn.addEventListener('click', () => {
        statsModal.classList.remove('hidden');
        loadABStats();
    });
    closeBtn.addEventListener('click', () => statsModal.classList.add('hidden'));
    statsModal.addEventListener('click', e => { if (e.target === statsModal) statsModal.classList.add('hidden'); });

    // ========================================================
    //  BUSINESS TAB — Copy buttons
    // ========================================================
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = document.getElementById(btn.dataset.target);
            if (!target) return;
            navigator.clipboard.writeText(target.innerText).then(() => {
                btn.textContent = '✅ Copied!';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = '📋 Copy';
                    btn.classList.remove('copied');
                }, 2000);
            });
        });
    });

    // ========================================================
    //  BUSINESS TAB — Live API Tester
    // ========================================================
    const bizTestBtn  = document.getElementById('biz-test-btn');
    const bizUserId   = document.getElementById('biz-user-id');
    const bizCount    = document.getElementById('biz-count');
    const bizResponse = document.getElementById('biz-response');
    const bizStatus   = document.getElementById('biz-response-status');

    if (bizTestBtn) {
        bizTestBtn.addEventListener('click', async () => {
            const uid = bizUserId.value.trim();
            const n   = parseInt(bizCount.value) || 5;
            if (!uid) { showToast('Enter a User ID'); return; }

            bizResponse.textContent = '⏳ Sending request...';
            bizStatus.textContent = '';
            bizStatus.className = 'status-pill';
            bizTestBtn.disabled = true;

            try {
                const t0   = performance.now();
                const resp = await fetch('/recommend', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: uid, n_recommendations: n, filter_purchased: true })
                });
                const ms   = (performance.now() - t0).toFixed(0);
                const data = await resp.json();

                bizStatus.textContent = `200 OK · ${ms}ms`;
                bizStatus.className = 'status-pill status-200';
                bizResponse.textContent = JSON.stringify(data, null, 2);

            } catch (err) {
                bizStatus.textContent = 'Error';
                bizStatus.className = 'status-pill status-err';
                bizResponse.textContent = `Error: ${err.message}`;
            } finally {
                bizTestBtn.disabled = false;
            }
        });
    }

    // ========================================================
    //  AI/ML TAB — Live Stats (initial load)
    // ========================================================
    async function loadLiveStats() {
        const set = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };
        try {
            const [healthResp, statsResp, abResp] = await Promise.all([
                fetch('/health'),
                fetch('/stats'),
                fetch('/ab-test/stats'),
            ]);
            const health = await healthResp.json();
            const stats  = await statsResp.json();
            const ab     = await abResp.json();

            set('ls-status', health.status === 'healthy' ? '✅ Healthy' : '⚠️ Degraded');
            set('ls-model',  health.model_loaded     ? '✅ Loaded'    : '❌ Not loaded');
            set('ls-redis',  health.redis_connected  ? '✅ Connected' : '❌ Offline');
            set('ls-users',        stats.n_users?.toLocaleString()        ?? '—');
            set('ls-items',        stats.n_items?.toLocaleString()        ?? '—');
            set('ls-interactions', stats.n_interactions?.toLocaleString() ?? '—');
            set('ls-sparsity',     stats.sparsity ?? '—');
            set('ls-v1-imp', ab.v1_impressions);
            set('ls-v2-imp', ab.v2_impressions);
            set('ls-v1-ctr', (ab.v1_ctr * 100).toFixed(2) + '%');
            set('ls-v2-ctr', (ab.v2_ctr * 100).toFixed(2) + '%');
            set('ls-winner', ab.winner || 'Need more data');

        } catch (err) {
            set('ls-status', '⚠️ Cannot reach API');
        }
    }

    const refreshStatsBtn = document.getElementById('refresh-stats-btn');
    if (refreshStatsBtn) {
        refreshStatsBtn.addEventListener('click', () => {
            showToast('Refreshing live stats...');
            loadLiveStats();
        });
    }

    // ========================================================
    //  AI/ML TAB — Dataset Upload
    // ========================================================
    const uploadZone   = document.getElementById('upload-zone');
    const fileInput    = document.getElementById('file-input');
    const browseBtn    = document.getElementById('browse-btn');
    const uploadResult = document.getElementById('upload-result');

    if (browseBtn) browseBtn.addEventListener('click', () => fileInput.click());

    if (uploadZone) {
        uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
        uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
        uploadZone.addEventListener('drop', e => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) handleUpload(file);
        });
        uploadZone.addEventListener('click', e => {
            if (e.target !== browseBtn) fileInput.click();
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files[0]) handleUpload(fileInput.files[0]);
        });
    }

    async function handleUpload(file) {
        if (!file.name.endsWith('.csv')) { showToast('Only CSV files are supported'); return; }
        uploadResult.className = 'upload-result';
        uploadResult.classList.remove('hidden');
        uploadResult.textContent = '⏳ Uploading...';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/dataset/upload', { method: 'POST', body: formData });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || 'Upload failed');

            uploadResult.className = 'upload-result success-card';
            uploadResult.innerHTML = `
                <div class="ur-header">
                    <span class="ur-check">✅</span>
                    <div>
                        <div class="ur-title">Dataset uploaded successfully!</div>
                        <div class="ur-sub">Your file has been saved and is ready for model re-training.</div>
                    </div>
                </div>
                <div class="ur-grid">
                    <div class="ur-cell"><div class="ur-label">📄 File Name</div><div class="ur-value">${data.filename}</div></div>
                    <div class="ur-cell"><div class="ur-label">📊 Total Rows</div><div class="ur-value">${data.rows_detected.toLocaleString()} records</div></div>
                    <div class="ur-cell"><div class="ur-label">📋 Columns Detected</div><div class="ur-value">${data.columns_detected.length} columns</div></div>
                    <div class="ur-cell"><div class="ur-label">💾 Saved To</div><div class="ur-value mono">${data.saved_to}</div></div>
                </div>
                <div class="ur-cols-list">${data.columns_detected.map(c => `<span class="col-chip">${c}</span>`).join('')}</div>
                <div class="ur-footer">🔄 The model pipeline can now be re-trained with this new dataset.</div>`;

        } catch (err) {
            uploadResult.className = 'upload-result error';
            uploadResult.textContent = `❌ Error: ${err.message}`;
        }
    }

    // ========================================================
    //  AI/ML TAB — Experiment Output Renderer
    // ========================================================
    const mlUserId = document.getElementById('ml-user-id');
    const v1Output = document.getElementById('v1-output');
    const v2Output = document.getElementById('v2-output');

    function renderExpOutput(container, data) {
        const CAT_ICONS = {
            'Audio':'🎧','Gaming':'🎮','Electronics':'⚡','Furniture':'🛋️',
            'Office Supplies':'📋','Technology':'💻','Decor':'🪴',
            'Appliances':'☕','Tablets':'📱','Wearables':'⌚',
            'Peripherals':'🖱️','Displays':'🖥️',
        };
        const scores   = data.recommendations.map(r => r.score);
        const maxScore = Math.max(...scores) || 1;
        const minScore = Math.min(...scores);
        const modelName = data.variant && data.variant.includes('v1')
            ? 'Collaborative Filtering (SVD)'
            : 'Hybrid AI (ALS + LightFM)';

        container.innerHTML = `
            <div class="exp-meta">
                <span class="exp-meta-chip">🧠 ${modelName}</span>
                <span class="exp-meta-chip">📦 ${data.recommendations.length} results</span>
                <span class="exp-meta-chip ${data.cached ? 'cached' : 'fresh'}">${data.cached ? '⚡ Cached' : '🔄 Fresh inference'}</span>
            </div>
            <div class="exp-explainer">
                <strong>How to read this:</strong> Each row is a product the AI thinks you would like.
                The <em>Confidence</em> bar shows how strongly the model believes in that recommendation
                — a longer bar means higher confidence.
            </div>
            ${data.recommendations.map(item => {
                const pname   = item.product_name || item.item_id;
                const cat     = item.category || 'Electronics';
                const icon    = CAT_ICONS[cat] || '🏷️';
                const normPct = maxScore === minScore
                    ? 100
                    : Math.round(((item.score - minScore) / (maxScore - minScore)) * 100);
                return `
                <div class="exp-item-v2">
                    <div class="exp-rank">#${item.rank}</div>
                    <div class="exp-icon">${icon}</div>
                    <div class="exp-details">
                        <div class="exp-name">${pname}</div>
                        <div class="exp-cat">${cat}</div>
                        <div class="exp-bar-wrap">
                            <div class="exp-bar-fill" style="width:${normPct}%"></div>
                        </div>
                    </div>
                    <div class="exp-conf">${normPct}<span class="exp-conf-unit">%</span></div>
                </div>`;
            }).join('')}`;
    }

    // ========================================================
    //  AI/ML TAB — Dataset Selection + Full Panel Updates
    // ========================================================

    const DS_META = {
        tech: {
            icon:  '⚡',
            label: 'Electronics & Tech',
            title: 'Electronics & Tech Dataset',
            sub:   'Showing Model Insights, Live State and Inference Experiment for this dataset',
            badge: 'Electronics & Tech',
        },
        superstore: {
            icon:  '🛒',
            label: 'Superstore & Furniture',
            title: 'Superstore & Furniture Dataset',
            sub:   'Showing Model Insights, Live State and Inference Experiment for this dataset',
            badge: 'Superstore & Furniture',
        },
        custom: {
            icon:  '📁',
            label: 'Custom CSV',
            title: 'Custom Uploaded Dataset',
            sub:   'Upload your own CSV below — required columns: user_id, item_id, rating',
            badge: 'Custom Dataset',
        },
    };

    let activeDataset = 'tech';

    const dsOptionBtns  = document.querySelectorAll('.ds-option-btn');
    const activeDsBadge = document.getElementById('active-ds-badge');
    const dsBannerIcon  = document.getElementById('ds-banner-icon');
    const dsBannerTitle = document.getElementById('ds-banner-title');
    const dsBannerSub   = document.getElementById('ds-banner-sub');
    const customUpload  = document.getElementById('custom-upload-area');
    const pipelineRaw   = document.getElementById('pipeline-raw-desc');

    // ---------- animated counter ----------
    function animateValue(el, target, isFloat) {
        if (!el) return;
        const duration = 600;
        const startTs  = performance.now();
        function step(ts) {
            const progress = Math.min((ts - startTs) / duration, 1);
            const eased    = 1 - Math.pow(1 - progress, 3);
            const current  = target * eased;
            el.textContent = isFloat ? current.toFixed(2) : Math.round(current).toLocaleString();
            if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    // ---------- banner flash ----------
    function flashBanner() {
        const banner = document.getElementById('ds-context-banner');
        if (!banner) return;
        banner.style.transition = 'none';
        banner.style.opacity    = '0';
        banner.style.transform  = 'translateY(-8px)';
        requestAnimationFrame(() => requestAnimationFrame(() => {
            banner.style.transition = 'opacity 0.35s ease, transform 0.35s ease';
            banner.style.opacity    = '1';
            banner.style.transform  = 'translateY(0)';
        }));
    }

    // ---------- Model Insights update ----------
    function updateModelInsights(metrics) {
        const v1 = metrics.v1 || {};
        animateValue(document.getElementById('v1-factors'),    v1.factors       || 100);
        animateValue(document.getElementById('v1-iterations'), v1.iterations    || 15);
        animateValue(document.getElementById('v1-rows'),       v1.training_rows || 0);
        animateValue(document.getElementById('v1-precision'),  v1.precision     || 0,    true);
        animateValue(document.getElementById('v1-recall'),     v1.recall        || 0,    true);
        animateValue(document.getElementById('v1-ndcg'),       v1.ndcg          || 0,    true);

        const v2Size = document.getElementById('v2-size');
        if (v2Size && metrics.v2_dataset_size) v2Size.textContent = metrics.v2_dataset_size;
    }

    // ---------- Live Stats update ----------
    function updateLiveStats(meta) {
        const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };

        set('ls-users',        meta.n_users?.toLocaleString()        ?? '—');
        set('ls-items',        meta.n_items?.toLocaleString()        ?? '—');
        set('ls-interactions', meta.n_interactions?.toLocaleString() ?? '—');
        set('ls-sparsity',     meta.sparsity                         ?? '—');

        Promise.all([fetch('/health'), fetch('/ab-test/stats')])
            .then(([hr, ar]) => Promise.all([hr.json(), ar.json()]))
            .then(([health, ab]) => {
                set('ls-status', health.status === 'healthy' ? '✅ Healthy' : '⚠️ Degraded');
                set('ls-model',  health.model_loaded    ? '✅ Loaded'    : '❌ Not loaded');
                set('ls-redis',  health.redis_connected ? '✅ Connected' : '❌ Offline');
                set('ls-v1-imp', ab.v1_impressions);
                set('ls-v2-imp', ab.v2_impressions);
                set('ls-v1-ctr', (ab.v1_ctr * 100).toFixed(2) + '%');
                set('ls-v2-ctr', (ab.v2_ctr * 100).toFixed(2) + '%');
                set('ls-winner', ab.winner || 'Need more data');
            })
            .catch(() => set('ls-status', '⚠️ Cannot reach API'));
    }

    // ---------- Experiment panel update ----------
    function updateExperimentPanel(sampleUserIds, ds) {
        if (mlUserId) {
            const exampleId = sampleUserIds && sampleUserIds.length > 0 ? sampleUserIds[0] : '';
            mlUserId.placeholder = `Enter a user_id from the ${DS_META[ds]?.label || ds} dataset`;
            mlUserId.value = exampleId;
        }
        if (v1Output) v1Output.innerHTML = `<div class="exp-empty">Dataset loaded — enter a User ID and run the model</div>`;
        if (v2Output) v2Output.innerHTML = `<div class="exp-empty">Dataset loaded — enter a User ID and run the model</div>`;
    }

    // ---------- Pandas Exploration update ----------
    async function updatePandasExploration(ds) {
        const container = document.getElementById('dataset-explore-container');
        if (!container) return;

        // Hide it by default when switching dataset
        container.classList.add('hidden');

        if (ds === 'custom') {
            container.innerHTML = `
                <h3 class="qr-title">Pandas DataFrame Exploration</h3>
                <div class="exp-empty">Pandas exploration is not available for custom datasets yet.</div>
            `;
            return;
        }

        container.innerHTML = `
            <h3 class="qr-title">Pandas DataFrame Exploration</h3>
            <div class="exp-loading"><div class="spinner-sm"></div> Running pandas functions on ${ds} dataset...</div>
        `;

        try {
            const resp = await fetch(`/dataset/explore?dataset=${ds}`);
            if (!resp.ok) throw new Error('API returned ' + resp.status);
            const data = await resp.json();

            if (!data.exploration || data.exploration.length === 0) {
                container.innerHTML = `
                    <h3 class="qr-title">Pandas DataFrame Exploration</h3>
                    <div class="exp-empty">No exploration data available.</div>
                `;
                return;
            }

            let html = '<h3 class="qr-title">Pandas DataFrame Exploration</h3>';
            data.exploration.forEach(exp => {
                html += `
                    <div class="pd-explore-block">
                        <div class="pd-explore-header">
                            <span class="pd-explore-method">${exp.code}</span>
                            <span class="pd-explore-purpose">${exp.purpose}</span>
                        </div>
                        <pre class="pd-explore-output">${exp.output}</pre>
                    </div>
                `;
            });
            container.innerHTML = html;
        } catch (err) {
            container.innerHTML = `
                <h3 class="qr-title">Pandas DataFrame Exploration</h3>
                <div class="exp-error">❌ Failed to load pandas exploration: ${err.message}</div>
            `;
        }
    }

    // ---------- Validated experiment runner ----------
    async function runExperimentWithValidation(variant, outputEl) {
        const uid = mlUserId ? mlUserId.value.trim() : '';
        if (!uid) {
            if (outputEl) outputEl.innerHTML = `<div class="exp-error">❌ Please enter a User ID before running the model.</div>`;
            return;
        }

        if (activeDataset === 'tech' && !uid.match(/^U0*(0?[1-9]|1[0-9]|20)$/)) {
            if (outputEl) outputEl.innerHTML = `
                <div class="exp-error">
                    ❌ <strong>Invalid User ID for Electronics &amp; Tech dataset.</strong><br>
                    Valid IDs are <code>U001</code> – <code>U020</code>.
                    <div class="exp-error-hint">💡 Try: U001, U005, U010, U015, U020</div>
                </div>`;
            return;
        }

        if (activeDataset === 'superstore' && !uid.match(/^U0*(2[1-9]|3[0-9]|40)$/)) {
            if (outputEl) outputEl.innerHTML = `
                <div class="exp-error">
                    ❌ <strong>Invalid User ID for Superstore &amp; Furniture dataset.</strong><br>
                    Valid IDs are <code>U021</code> – <code>U040</code>.
                    <div class="exp-error-hint">💡 Try: U021, U025, U030, U035, U040</div>
                </div>`;
            return;
        }

        if (outputEl) outputEl.innerHTML = `<div class="exp-loading"><div class="spinner-sm"></div> Running inference…</div>`;

        try {
            const resp = await fetch(`/recommend/${variant}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: uid, n_recommendations: 10, filter_purchased: true })
            });
            const data = await resp.json();
            if (!resp.ok) {
                outputEl.innerHTML = `<div class="exp-error">❌ ${data.detail || data.error || `Error ${resp.status}`}</div>`;
                return;
            }
            renderExpOutput(outputEl, data);
        } catch (err) {
            if (outputEl) outputEl.innerHTML = `<div class="exp-error">❌ Network error: ${err.message}</div>`;
        }
    }

    // ---------- Main selectDataset ----------
    async function selectDataset(ds) {
        activeDataset = ds;
        const meta = DS_META[ds] || DS_META.tech;

        dsOptionBtns.forEach(b => b.classList.toggle('active', b.dataset.ds === ds));

        if (activeDsBadge) activeDsBadge.textContent = meta.badge;
        if (dsBannerIcon)  dsBannerIcon.textContent  = meta.icon;
        if (dsBannerTitle) dsBannerTitle.textContent  = meta.title;
        if (dsBannerSub)   dsBannerSub.textContent    = meta.sub;
        flashBanner();

        if (customUpload) customUpload.classList.toggle('hidden', ds !== 'custom');

        if (ds === 'custom') {
            showToast('Upload your CSV below to use custom data');
            return;
        }

        showToast(`Switching to: ${meta.label}…`);

        try {
            const resp = await fetch(`/dataset/meta?dataset=${ds}`);
            if (!resp.ok) throw new Error(`API returned ${resp.status}`);
            const data = await resp.json();

            if (pipelineRaw) pipelineRaw.textContent = data.metrics?.pipeline_raw || meta.label;

            const v2Size = document.getElementById('v2-size');
            if (v2Size) v2Size.textContent = `${data.n_interactions} interactions`;

            if (data.metrics && data.metrics.v1) {
                data.metrics.v1.training_rows = data.n_interactions;
            }

            updateModelInsights(data.metrics || {});
            updateLiveStats(data);
            updateExperimentPanel(data.sample_user_ids, ds);
            updatePandasExploration(ds);

            showToast(`✅ ${meta.label} — ${data.n_interactions} interactions · ${data.n_users} users`);
        } catch (err) {
            showToast(`⚠️ Could not load metadata: ${err.message}`);
        }
    }

    // Attach toggle listener
    const toggleStatsBtn = document.getElementById('toggle-stats-btn');
    const exploreContainer = document.getElementById('dataset-explore-container');
    if (toggleStatsBtn && exploreContainer) {
        toggleStatsBtn.addEventListener('click', () => {
            exploreContainer.classList.toggle('hidden');
            if (!exploreContainer.classList.contains('hidden')) {
                exploreContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    }

    // Attach dataset button click listeners
    dsOptionBtns.forEach(btn => {
        btn.addEventListener('click', () => selectDataset(btn.dataset.ds));
    });

    // Attach run button listeners
    const runV1BtnEl = document.getElementById('run-v1-btn');
    const runV2BtnEl = document.getElementById('run-v2-btn');
    if (runV1BtnEl) runV1BtnEl.addEventListener('click', () => runExperimentWithValidation('v1', v1Output));
    if (runV2BtnEl) runV2BtnEl.addEventListener('click', () => runExperimentWithValidation('v2', v2Output));

});