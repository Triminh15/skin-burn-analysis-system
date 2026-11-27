/**
 * BurnScan - Interactive Caliper & Anatomical Calibration Module (Step 2)
 * Hỗ trợ 2 chế độ chuẩn Y khoa:
 * 1. Chuẩn hóa Giải phẫu AI (Rule of Nines / Palmar 1% + Tách vùng da tự động)
 * 2. Thước chuẩn Vật lý (Kéo thước Caliper quy đổi sang độ phân giải ảnh gốc thực tế)
 */

class CaliperManager {
  static init() {
    this.knob1 = document.getElementById('knob-1');
    this.knob2 = document.getElementById('knob-2');
    this.stage = document.getElementById('calib-stage-wrapper');

    if (!this.knob1 || !this.knob2 || !this.stage) return;

    this.knob1.addEventListener('pointerdown', (e) => this.onPointerDown(e, 'knob1'));
    this.knob2.addEventListener('pointerdown', (e) => this.onPointerDown(e, 'knob2'));

    this.setCalibrationMode(appState.calibrationMode || 'anatomical', true);
    this.refreshAnatomicalReadout();
    this.updateGraphics();
  }

  static onPointerDown(e, knobKey) {
    e.preventDefault();
    appState.isDraggingKnob = knobKey;

    const onPointerMove = (evt) => {
      if (!appState.isDraggingKnob) return;
      const rect = this.stage.getBoundingClientRect();
      const x = Math.max(10, Math.min(rect.width - 10, evt.clientX - rect.left));
      const y = Math.max(10, Math.min(rect.height - 10, evt.clientY - rect.top));

      if (appState.isDraggingKnob === 'knob1') {
        appState.knob1Pos = { x, y };
      } else if (appState.isDraggingKnob === 'knob2') {
        appState.knob2Pos = { x, y };
      }
      this.updateGraphics();
    };

    const onPointerUp = () => {
      appState.isDraggingKnob = null;
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
  }

  /**
   * Tính hệ số phóng đại giữa kích thước ảnh hiển thị CSS (object-contain) và ảnh gốc (naturalWidth)
   */
  static getImageScaleFactor() {
    const img = document.getElementById('img-calib-canvas');
    if (!img || !img.naturalWidth || !img.naturalHeight || !this.stage) {
      return 1.0;
    }
    const stageW = this.stage.clientWidth || 500;
    const stageH = this.stage.clientHeight || 480;
    const imgRatio = img.naturalWidth / img.naturalHeight;
    const stageRatio = stageW / stageH;

    let renderedW = stageW;
    if (imgRatio > stageRatio) {
      renderedW = stageW;
    } else {
      renderedW = stageH * imgRatio;
    }
    return renderedW > 0 ? (img.naturalWidth / renderedW) : 1.0;
  }

  static updateGraphics() {
    if (!this.knob1 || !this.knob2) return;

    this.knob1.style.left = `${appState.knob1Pos.x}px`;
    this.knob1.style.top = `${appState.knob1Pos.y}px`;
    this.knob2.style.left = `${appState.knob2Pos.x}px`;
    this.knob2.style.top = `${appState.knob2Pos.y}px`;

    const line = document.getElementById('svg-calip-line');
    if (line) {
      line.setAttribute('x1', appState.knob1Pos.x);
      line.setAttribute('y1', appState.knob1Pos.y);
      line.setAttribute('x2', appState.knob2Pos.x);
      line.setAttribute('y2', appState.knob2Pos.y);
    }

    const tick1 = document.getElementById('svg-calip-tick1');
    if (tick1) {
      tick1.setAttribute('x1', appState.knob1Pos.x);
      tick1.setAttribute('y1', appState.knob1Pos.y - 12);
      tick1.setAttribute('x2', appState.knob1Pos.x);
      tick1.setAttribute('y2', appState.knob1Pos.y + 12);
    }

    const tick2 = document.getElementById('svg-calip-tick2');
    if (tick2) {
      tick2.setAttribute('x1', appState.knob2Pos.x);
      tick2.setAttribute('y1', appState.knob2Pos.y - 12);
      tick2.setAttribute('x2', appState.knob2Pos.x);
      tick2.setAttribute('y2', appState.knob2Pos.y + 12);
    }

    // Khoảng cách trên màn hình và quy đổi sang pixel ảnh gốc (natural pixels)
    const dx = appState.knob2Pos.x - appState.knob1Pos.x;
    const dy = appState.knob2Pos.y - appState.knob1Pos.y;
    const cssDistPx = Math.max(4, Math.sqrt(dx * dx + dy * dy));
    const scale = this.getImageScaleFactor();

    appState.calibDistPx = Math.round(cssDistPx);
    appState.calibNaturalDistPx = Math.max(4, Math.round(cssDistPx * scale));

    const distLabel = document.getElementById('calib-pixel-dist');
    if (distLabel) {
      if (appState.calibrationMode === 'ruler') {
        distLabel.innerText = `Thước: ${appState.calibNaturalDistPx} px (ảnh gốc)`;
      } else {
        distLabel.innerText = 'Chế độ: Tách vùng da & Giải phẫu AI';
      }
    }

    this.recomputeRatio();
  }

  static recomputeRatio() {
    const inputDim = document.getElementById('input-real-dimension');
    const realVal = inputDim ? parseFloat(inputDim.value) : 8.56;

    if (realVal && realVal > 0 && appState.calibNaturalDistPx > 0) {
      appState.calibRealCm = realVal;
      appState.calibPxPerCm = parseFloat((appState.calibNaturalDistPx / realVal).toFixed(2));

      const ratioLabel = document.getElementById('label-derived-ratio');
      const ratioPill = document.getElementById('calib-ratio-pill');

      if (ratioLabel) ratioLabel.innerText = `${appState.calibPxPerCm.toFixed(1).replace('.', ',')} px/cm`;
      if (ratioPill) {
        ratioPill.innerText = appState.calibrationMode === 'ruler'
          ? `${appState.calibPxPerCm.toFixed(1)} px/cm`
          : 'Rule of Nines + Skin Ratio';
      }
    }
  }

  static setCalibrationMode(mode, silent = false) {
    appState.calibrationMode = mode === 'ruler' ? 'ruler' : 'anatomical';

    const btnAnat = document.getElementById('btn-calib-mode-anatomical');
    const btnRuler = document.getElementById('btn-calib-mode-ruler');
    const overlay = document.getElementById('caliper-overlay');
    const rulerPanel = document.getElementById('ruler-presets-panel');
    const anatPanel = document.getElementById('anatomical-explain-panel');
    const rulerTools = document.getElementById('ruler-toolbar-btns');
    const bottomHint = document.getElementById('calib-bottom-hint');
    const anatRightCard = document.getElementById('anatomical-selector-card');
    const rulerRightCard = document.getElementById('ruler-mode-right-card');

    if (appState.calibrationMode === 'ruler') {
      if (btnRuler) btnRuler.className = 'flex-1 py-2 px-3 rounded-lg bg-white text-blue-700 font-semibold text-[12px] shadow-xs transition-all flex items-center justify-center gap-1.5';
      if (btnAnat) btnAnat.className = 'flex-1 py-2 px-3 rounded-lg text-gray-600 hover:text-black font-medium text-[12px] transition-all flex items-center justify-center gap-1.5';
      overlay?.classList.remove('hidden');
      rulerPanel?.classList.remove('hidden');
      rulerTools?.classList.remove('hidden');
      anatPanel?.classList.add('hidden');
      anatRightCard?.classList.add('hidden');
      rulerRightCard?.classList.remove('hidden');
      if (bottomHint) bottomHint.innerText = 'Kéo hai núm vàng vào hai đầu vật chuẩn có thật trong ảnh (CCCD / Thước y tế / Đồng xu)';
      if (!silent) showToast('Chế độ Thước chuẩn vật lý: Tự tính cm² → % TBSA, không cần chọn vùng giải phẫu');
    } else {
      if (btnAnat) btnAnat.className = 'flex-1 py-2 px-3 rounded-lg bg-white text-blue-700 font-semibold text-[12px] shadow-xs transition-all flex items-center justify-center gap-1.5';
      if (btnRuler) btnRuler.className = 'flex-1 py-2 px-3 rounded-lg text-gray-600 hover:text-black font-medium text-[12px] transition-all flex items-center justify-center gap-1.5';
      overlay?.classList.add('hidden');
      rulerPanel?.classList.add('hidden');
      rulerTools?.classList.add('hidden');
      anatPanel?.classList.remove('hidden');
      anatRightCard?.classList.remove('hidden');
      rulerRightCard?.classList.add('hidden');
      if (bottomHint) bottomHint.innerText = 'AI sẽ tự tách diện tích vùng da bộ phận khỏi phông nền và quy đổi theo Luật số 9 / Bàn tay 1%';
      if (!silent) showToast('Chế độ Chuẩn hóa Giải phẫu (Rule of Nines & Tỷ lệ Vết bỏng/Vùng da)');
    }

    this.updateGraphics();
  }

  static setPreset(val, name) {
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.classList.remove('active', 'bg-blue-50', 'border-blue-500');
    });

    const activeBtn = Array.from(document.querySelectorAll('.preset-btn')).find(
      btn => btn.dataset.val === (val ? String(val) : 'custom')
    );
    if (activeBtn) activeBtn.classList.add('active', 'bg-blue-50', 'border-blue-500');

    if (val !== null) {
      const inputDim = document.getElementById('input-real-dimension');
      if (inputDim) inputDim.value = val;
      appState.calibRealCm = val;
    }
    this.recomputeRatio();
  }

  static reset() {
    const w = this.stage?.clientWidth || 500;
    const h = this.stage?.clientHeight || 480;
    appState.knob1Pos = { x: Math.round(w * 0.25), y: Math.round(h * 0.5) };
    appState.knob2Pos = { x: Math.round(w * 0.75), y: Math.round(h * 0.5) };
    this.updateGraphics();
    showToast('Đã đặt lại vị trí thước đo');
  }

  static toggleVisibility() {
    const overlay = document.getElementById('caliper-overlay');
    const btn = document.getElementById('btn-toggle-caliper');
    if (!overlay || !btn) return;

    if (overlay.classList.contains('hidden')) {
      overlay.classList.remove('hidden');
      btn.classList.add('bg-white/20');
    } else {
      overlay.classList.add('hidden');
      btn.classList.remove('bg-white/20');
    }
  }

  static setBodyPart(val) {
    const select = document.getElementById('select-body-part');
    if (select) select.value = val;
    appState.bodyPart = val;

    document.querySelectorAll('.region-chip-btn').forEach(btn => {
      if (btn.dataset.part === val) {
        btn.className = 'region-chip-btn px-2.5 py-1 rounded-lg text-[12px] bg-blue-50 text-blue-700 border border-blue-300 font-semibold transition-all';
      } else {
        btn.className = 'region-chip-btn px-2.5 py-1 rounded-lg text-[12px] bg-gray-100 text-gray-700 hover:bg-gray-200 font-medium transition-all';
      }
    });

    this.refreshAnatomicalReadout();
    const info = BODY_PARTS[val];
    if (info) {
      showToast(`Đã chọn vùng: ${info.name} (${info.pct}% TBSA)`);
    }
  }

  static setRegionCoverage(cov) {
    const numCov = Math.max(0.05, Math.min(1.0, parseFloat(cov) || 1.0));
    appState.regionCoverage = numCov;

    document.querySelectorAll('.cov-btn').forEach(btn => {
      if (Math.abs(parseFloat(btn.dataset.cov) - numCov) < 0.01) {
        btn.className = 'cov-btn py-1.5 px-2 rounded-lg text-[12px] bg-blue-600 text-white font-semibold shadow-xs transition-all text-center';
      } else {
        btn.className = 'cov-btn py-1.5 px-2 rounded-lg text-[12px] bg-gray-100 text-gray-700 hover:bg-gray-200 font-medium transition-all text-center';
      }
    });

    this.refreshAnatomicalReadout();
  }

  static refreshAnatomicalReadout() {
    const info = BODY_PARTS[appState.bodyPart] || { name: appState.bodyPart, pct: 9.0 };
    const cov = appState.regionCoverage || 1.0;
    const maxVisible = (info.pct * cov);
    const maxStr = maxVisible < 1.0 ? maxVisible.toFixed(2).replace('.', ',') : maxVisible.toFixed(1).replace('.', ',');

    const el = document.getElementById('label-effective-max-tbsa');
    if (el) {
      el.innerText = `${info.name} (${info.pct}%) × ${Math.round(cov * 100)}% khung vùng → Trần tối đa ${maxStr}% TBSA`;
    }
  }
}
