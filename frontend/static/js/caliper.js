/**
 * BurnScan - Interactive Caliper & Calibration Module (Step 2)
 * Kéo thả thước đo caliper tương tác và quy đổi px/cm
 */

class CaliperManager {
  static init() {
    this.knob1 = document.getElementById('knob-1');
    this.knob2 = document.getElementById('knob-2');
    this.stage = document.getElementById('calib-stage-wrapper');

    if (!this.knob1 || !this.knob2 || !this.stage) return;

    this.knob1.addEventListener('pointerdown', (e) => this.onPointerDown(e, 'knob1'));
    this.knob2.addEventListener('pointerdown', (e) => this.onPointerDown(e, 'knob2'));

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

    // Distance in pixels
    const dx = appState.knob2Pos.x - appState.knob1Pos.x;
    const dy = appState.knob2Pos.y - appState.knob1Pos.y;
    appState.calibDistPx = Math.max(10, Math.round(Math.sqrt(dx * dx + dy * dy)));

    const distLabel = document.getElementById('calib-pixel-dist');
    if (distLabel) distLabel.innerText = `Độ dài: ${appState.calibDistPx} px`;

    this.recomputeRatio();
  }

  static recomputeRatio() {
    const inputDim = document.getElementById('input-real-dimension');
    const realVal = inputDim ? parseFloat(inputDim.value) : 8.56;

    if (realVal && realVal > 0 && appState.calibDistPx > 0) {
      appState.calibRealCm = realVal;
      appState.calibPxPerCm = parseFloat((appState.calibDistPx / realVal).toFixed(1));

      const ratioLabel = document.getElementById('label-derived-ratio');
      const ratioPill = document.getElementById('calib-ratio-pill');

      if (ratioLabel) ratioLabel.innerText = `${appState.calibPxPerCm.toString().replace('.', ',')} px/cm`;
      if (ratioPill) ratioPill.innerText = `${appState.calibPxPerCm} px/cm`;
    }
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
    const info = BODY_PARTS[val];
    if (info) {
      showToast(`Đã chọn vùng: ${info.name} (${info.pct}% TBSA)`);
    }
  }
}

