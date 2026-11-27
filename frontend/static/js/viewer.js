/**
 * BurnScan - Medical Stage & Results Viewer Module (Step 4)
 * Trực quan hóa hình ảnh y khoa, thanh trượt so sánh & kết quả lâm sàng
 */

class ResultsViewer {
  /**
   * Cập nhật vị trí thanh trượt chia đôi Split Slider
   */
  static onSplitSliderChange(val) {
    const pane = document.getElementById('slider-mask-pane');
    const divider = document.getElementById('slider-divider');
    if (pane && divider) {
      pane.style.width = `${val}%`;
      divider.style.left = `${val}%`;
    }
  }

  /**
   * Chuyển đổi các chế độ xem hiển thị trên Viewport
   * (split, overlay, contours, roi, orig)
   */
  static setDisplayMode(mode) {
    const splitCont = document.getElementById('mode-split-container');
    const staticImg = document.getElementById('img-static-viewer');
    const imgs = appState.activeSessionData?.images || {};

    document.querySelectorAll('#viewer-container button').forEach(b => {
      b.classList.remove('bg-white/25', 'font-semibold');
    });
    const activeBtn = document.getElementById(`btn-mode-${mode}`);
    if (activeBtn) activeBtn.classList.add('bg-white/25', 'font-semibold');

    if (mode === 'split') {
      splitCont?.classList.remove('hidden');
      staticImg?.classList.add('hidden');
    } else {
      splitCont?.classList.add('hidden');
      staticImg?.classList.remove('hidden');
      if (mode === 'overlay') staticImg.src = imgs.overlay || imgs.original;
      else if (mode === 'contours') staticImg.src = imgs.contours || imgs.original;
      else if (mode === 'roi') staticImg.src = imgs.roi || imgs.original;
      else if (mode === 'orig') staticImg.src = imgs.original;
    }
  }

  /**
   * Chuyển tab trong màn hình Kết quả (5 tabs Apple HIG)
   */
  static switchResultTab(tabName) {
    document.querySelectorAll('.res-tab-btn').forEach(b => {
      b.className = 'res-tab-btn flex-1 py-2 px-3 rounded-lg text-gray-600 hover:text-black font-medium text-[13px] text-center transition-all whitespace-nowrap';
    });
    document.querySelectorAll('.res-tab-content').forEach(c => c.classList.add('hidden'));

    const activeBtn = document.getElementById(`tab-btn-${tabName === 'segmentation' ? 'seg' : tabName}`);
    if (activeBtn) {
      activeBtn.className = 'res-tab-btn active flex-1 py-2 px-3 rounded-lg bg-white text-black font-semibold text-[13px] shadow-xs text-center transition-all whitespace-nowrap';
    }

    const activeContent = document.getElementById(`res-tab-${tabName}`);
    if (activeContent) activeContent.classList.remove('hidden');
  }

  /**
   * Đổ dữ liệu phân tích từ Backend lên toàn bộ giao diện Bước 4
   */
  static populateResults(data) {
    const p = data.patient || {};
    const m = data.metrics || {};
    const c = data.classification || {};
    const imgs = data.images || {};
    const pk = m.parkland_plan || null;

    // Hàm định dạng % TBSA chuẩn chữ số có nghĩa (tránh 0.03% bị làm tròn thành 0.0%)
    const formatTBSA = (val) => {
      const num = Number(val) || 0.0;
      if (num === 0) return '0,0';
      if (num < 0.01) return num.toFixed(3).replace('.', ',');
      if (num < 1.0) return num.toFixed(2).replace('.', ',');
      return num.toFixed(1).replace('.', ',');
    };

    // 1. Tóm tắt bệnh nhân trên Header
    const nameEl = document.getElementById('res-patient-name');
    const metaEl = document.getElementById('res-patient-meta');
    const codeEl = document.getElementById('res-session-code');
    const timeEl = document.getElementById('res-timestamp');

    if (nameEl) nameEl.innerText = p.name || 'Bệnh nhân vãng lai';
    const heightPart = p.height_cm ? ` · ${p.height_cm} cm` : '';
    if (metaEl) metaEl.innerText = `${p.gender || 'Nam'} · ${p.age || 30} tuổi${heightPart} · ${p.weight_kg || 65} kg`;
    if (codeEl) codeEl.innerText = data.session_id || 'BS-2024';
    if (timeEl && data.created_at) timeEl.innerText = data.created_at;

    // 2. Huy hiệu cấp độ bỏng & Mô tả bệnh học
    const deg = c.predicted_degree || 'Độ 1';
    const degBadge = document.getElementById('badge-burn-degree-header');
    const degHero = document.getElementById('badge-degree-hero');
    const degTitle = document.getElementById('card-degree-title');
    const pathDetail = document.getElementById('text-pathology-detail');
    const healTime = document.getElementById('text-healing-time');
    const scarRisk = document.getElementById('text-scar-risk');
    const painLevel = document.getElementById('text-pain-level');

    if (degBadge) degBadge.innerText = deg;
    if (degTitle) degTitle.innerText = deg;

    if (deg.includes('Độ 1') || deg.includes('Superficial')) {
      if (degBadge) degBadge.className = 'px-2.5 py-0.5 rounded-full text-[12px] font-bold bg-green-100 text-green-800 border border-green-300';
      if (degHero) {
        degHero.className = 'w-14 h-14 rounded-2xl bg-green-100 text-green-700 flex items-center justify-center font-bold text-[22px] shadow-xs';
        degHero.innerText = 'I';
      }
      if (pathDetail) pathDetail.innerText = 'Tổn thương nông tầng thượng bì (Epidermis). Da đỏ xung huyết, khô, không có bọng nước, đau rát nhẹ, tự lành sau 3-6 ngày không để lại sẹo.';
      if (healTime) healTime.innerText = '3 - 6 ngày';
      if (scarRisk) scarRisk.innerText = 'Không để lại sẹo';
      if (painLevel) painLevel.innerText = 'Rát nhẹ, châm chích';
    } else if (deg.includes('Độ 2') || deg.includes('Partial')) {
      if (degBadge) degBadge.className = 'px-2.5 py-0.5 rounded-full text-[12px] font-bold bg-orange-100 text-orange-800 border border-orange-300';
      if (degHero) {
        degHero.className = 'w-14 h-14 rounded-2xl bg-orange-100 text-orange-600 flex items-center justify-center font-bold text-[22px] shadow-xs';
        degHero.innerText = 'II';
      }
      if (pathDetail) pathDetail.innerText = 'Tổn thương lớp trung bì (Dermis) với nhiều bọng nước chứa dịch trong/vàng chanh. Nền vết bỏng hồng ướt, tăng cảm giác đau buốt dữ dội khi tiếp xúc.';
      if (healTime) healTime.innerText = '10 - 21 ngày';
      if (scarRisk) scarRisk.innerText = 'Trung bình (Nguy cơ sẹo nếu nhiễm trùng)';
      if (painLevel) painLevel.innerText = 'Đau buốt dữ dội';
    } else {
      if (degBadge) degBadge.className = 'px-2.5 py-0.5 rounded-full text-[12px] font-bold bg-red-100 text-red-800 border border-red-300';
      if (degHero) {
        degHero.className = 'w-14 h-14 rounded-2xl bg-red-100 text-red-600 flex items-center justify-center font-bold text-[22px] shadow-xs';
        degHero.innerText = 'III';
      }
      if (pathDetail) pathDetail.innerText = 'Tổn thương toàn bộ lớp biểu bì, trung bì và hạ bì. Bề mặt da khô cứng màu trắng bệch hoặc cháy đen hoại tử than hóa. Mất cảm giác đau tại tâm tổn thương do đầu dây thần kinh bị hủy hoại hoàn toàn.';
      if (healTime) healTime.innerText = '> 1 - 2 tháng (Cần ghép da)';
      if (scarRisk) scarRisk.innerText = 'Rất cao (Sẹo lồi & co kéo biến dạng)';
      if (painLevel) painLevel.innerText = 'Mất cảm giác đau (Gây sốc bỏng)';
    }

    // 3. Cảnh báo cấp cứu Triage
    const urgentBanner = document.getElementById('urgent-triage-banner');
    const urgentTitle = document.getElementById('urgent-triage-title');
    const urgentDesc = document.getElementById('urgent-triage-desc');
    const isDegree3 = deg.includes('Độ 3') || deg.includes('Full Thickness');
    const isShockRisk = (m.estimated_tbsa_pct >= 10.0) || (pk && pk.urgent_triage);

    if (urgentBanner) {
      if (isShockRisk) {
        urgentBanner.className = 'rounded-2xl bg-red-50 border border-red-200 p-4 shadow-xs';
        if (urgentTitle) urgentTitle.innerText = 'CẢNH BÁO TRIAGE: NGUY CƠ SỐC BỎNG & CHỈ ĐỊNH BÙ DỊCH KHẨN CẤP';
        if (urgentDesc) urgentDesc.innerText = `Diện tích tổn thương đạt ${formatTBSA(m.estimated_tbsa_pct)}% TBSA. Cần thiết lập ngay 2 đường truyền tĩnh mạch lớn Ringer Lactate, đặt ống thông tiểu theo dõi và chuẩn bị chuyển viện chuyên khoa Bỏng.`;
        urgentBanner.classList.remove('hidden');
      } else if (isDegree3) {
        urgentBanner.className = 'rounded-2xl bg-amber-50 border border-amber-300 p-4 shadow-xs';
        if (urgentTitle) urgentTitle.innerText = 'CẢNH BÁO LÂM SÀNG: TỔN THƯƠNG HOẠI TỬ ĐỘ 3 (CẦN XỬ TRÍ NGOẠI KHOA)';
        if (urgentDesc) urgentDesc.innerText = `Diện tích tổn thương nhỏ (${formatTBSA(m.estimated_tbsa_pct)}% TBSA) nên KHÔNG CẦN truyền dịch chống sốc. Tuy nhiên, vì là Bỏng Độ 3 (hoại tử mô sâu), bệnh nhân bắt buộc phải đến cơ sở y tế để bác sĩ cắt lọc mô chết (Debridement) và tiêm phòng uốn ván.`;
        urgentBanner.classList.remove('hidden');
      } else {
        urgentBanner.classList.add('hidden');
      }
    }

    // 4. Tab 1 - Phân đoạn ảnh & Thước đo định lượng
    const origImg = document.getElementById('img-slider-orig');
    const overlayImg = document.getElementById('img-slider-overlay');
    const staticImg = document.getElementById('img-static-viewer');

    if (origImg) origImg.src = imgs.original || '';
    if (overlayImg) overlayImg.src = imgs.overlay || imgs.original || '';
    if (staticImg) staticImg.src = imgs.overlay || '';
    this.onSplitSliderChange(50);

    const burnPxEl = document.getElementById('metric-burn-pixels');
    const frameRatioEl = document.getElementById('metric-frame-ratio');
    const burnSkinRatioEl = document.getElementById('metric-burn-skin-ratio');
    const skinInFrameHintEl = document.getElementById('metric-skin-in-frame-hint');
    const calibHintEl = document.getElementById('metric-calib-hint');
    const areaCm2El = document.getElementById('metric-area-cm2');
    const tbsaPctEl = document.getElementById('metric-tbsa-pct');
    const tbsaFormulaEl = document.getElementById('metric-tbsa-formula');
    const resDimEl = document.getElementById('viewer-res-dim');
    const calibBadgeEl = document.getElementById('badge-calib-mode');
    const framingWarnBox = document.getElementById('box-framing-warning');
    const framingWarnText = document.getElementById('text-framing-warning');

    if (resDimEl && m.image_dimensions) {
      resDimEl.innerText = `${m.image_dimensions.replace('x', ' × ')} px`;
    }
    if (calibBadgeEl) {
      calibBadgeEl.innerText = m.calibration_mode === 'ruler' ? 'Thước chuẩn px/cm' : (m.atlas_method || 'Rule of Nines AI');
    }

    const pctFrame = Number(m.pct_frame_area || 0);
    const pctBurnSkin = Number(m.pct_burn_of_skin !== undefined ? m.pct_burn_of_skin : pctFrame);
    const pctSkinFrame = Number(m.pct_skin_in_frame !== undefined ? m.pct_skin_in_frame : 100);

    if (burnPxEl) burnPxEl.innerText = `${(m.burn_pixels || 0).toLocaleString()} px`;
    if (frameRatioEl) frameRatioEl.innerText = `${pctFrame.toFixed(1).replace('.', ',')} %`;
    if (burnSkinRatioEl) burnSkinRatioEl.innerText = `${pctBurnSkin.toFixed(1).replace('.', ',')} %`;
    if (skinInFrameHintEl) {
      skinInFrameHintEl.innerText = pctSkinFrame < 98.0
        ? `Đã tách phông nền (Vùng da chiếm ${pctSkinFrame.toFixed(1).replace('.', ',')}% ảnh)`
        : 'Vùng da chiếm trọn khung hình ảnh';
    }

    const rowBurnSkin = document.getElementById('row-metric-burn-skin');
    const rowAreaCm2 = document.getElementById('row-metric-area-cm2');
    const isRulerMode = (m.calibration_mode === 'ruler' && Boolean(m.area_cm2));

    if (isRulerMode) {
      rowBurnSkin?.classList.add('hidden');
      rowAreaCm2?.classList.remove('hidden');
      if (areaCm2El) {
        const sideEq = Math.sqrt(Number(m.area_cm2)).toFixed(1).replace('.', ',');
        const bsaCm2 = m.bsa_cm2 ? Math.round(Number(m.bsa_cm2)).toLocaleString() : '17.000';
        areaCm2El.innerText = `${Number(m.area_cm2).toFixed(1).replace('.', ',')} cm²`;
        if (calibHintEl) {
          calibHintEl.innerText = `Cỡ ~${sideEq}×${sideEq} cm · Tổng da cơ thể (BSA): ${bsaCm2} cm²`;
        }
      }
    } else {
      rowBurnSkin?.classList.remove('hidden');
      rowAreaCm2?.classList.add('hidden');
    }

    if (tbsaPctEl) tbsaPctEl.innerText = `${formatTBSA(m.estimated_tbsa_pct)} %`;

    const regionInfo = BODY_PARTS[m.body_part] || {
      name: m.body_part_name || m.body_part,
      pct: m.region_weight_pct || 9.0
    };
    const regWeight = m.region_weight_pct !== undefined ? m.region_weight_pct : regionInfo.pct;
    const regName = m.body_part_name || regionInfo.name;
    const regCov = Number(m.region_coverage || 1.0);

    const partNameEl = document.getElementById('metric-body-part-name');
    const regionTitleEl = document.getElementById('card-region-title');
    const regionMaxEl = document.getElementById('card-region-max-tbsa');

    if (m.calibration_mode === 'ruler' && m.area_cm2) {
      const bsaM2 = m.bsa_cm2 ? (Number(m.bsa_cm2) / 10000.0).toFixed(2).replace('.', ',') : '1,70';
      if (partNameEl) partNameEl.innerText = 'Theo Thước chuẩn Vật lý & BSA Mosteller';
      if (tbsaFormulaEl) {
        tbsaFormulaEl.innerText = m.formula_summary || `${Number(m.area_cm2).toFixed(1).replace('.', ',')} cm² / BSA (${bsaM2} m²) = ${formatTBSA(m.estimated_tbsa_pct)}% TBSA`;
      }
      if (regionTitleEl) regionTitleEl.innerText = 'Phương pháp: Đo trực tiếp bằng Thước chuẩn';
      if (regionMaxEl) regionMaxEl.innerText = `BSA: ${bsaM2} m²`;
    } else {
      if (partNameEl) partNameEl.innerText = `${m.atlas_method || 'Luật số 9'} (${regName}: ${regWeight}%)`;
      if (tbsaFormulaEl) {
        tbsaFormulaEl.innerText = m.formula_summary || `${pctBurnSkin.toFixed(1).replace('.', ',')}% vùng da × ${regWeight}%${regCov < 0.99 ? ` × ${Math.round(regCov * 100)}%` : ''} = ${formatTBSA(m.estimated_tbsa_pct)}% TBSA`;
      }
      if (regionTitleEl) regionTitleEl.innerText = `Vùng giải phẫu: ${regName}`;
      if (regionMaxEl) regionMaxEl.innerText = `Trần ${regWeight}% TBSA`;
    }

    if (framingWarnBox && framingWarnText) {
      if (m.framing_warning) {
        framingWarnText.innerText = m.framing_warning;
        framingWarnBox.classList.remove('hidden');
      } else {
        framingWarnBox.classList.add('hidden');
      }
    }

    // 5. Tab 2 - Phân loại độ sâu & Xác suất Softmax
    const conf = c.confidence || 0.0;
    const isAbstain = (conf < 60.0) || Boolean(c.is_abstaining);
    const confEl = document.getElementById('metric-confidence');
    const barConfEl = document.getElementById('bar-confidence');
    if (confEl) {
      confEl.innerText = `${conf.toFixed(1).replace('.', ',')} %`;
      confEl.className = isAbstain ? 'font-bold text-red-600 tabular-nums' : 'font-bold text-orange-600 tabular-nums';
    }
    if (barConfEl) {
      barConfEl.style.width = `${Math.min(100, conf)}%`;
      barConfEl.className = isAbstain ? 'h-full bg-red-500 rounded-full' : 'h-full bg-orange-500 rounded-full';
    }

    // Toggle chốt chặn đạo đức y tế (Low Confidence Abstention)
    const lowConfBox = document.getElementById('box-low-confidence-warning');
    const tab5AbstainBanner = document.getElementById('tab5-abstention-banner');
    if (lowConfBox) {
      if (isAbstain) lowConfBox.classList.remove('hidden');
      else lowConfBox.classList.add('hidden');
    }
    if (tab5AbstainBanner) {
      if (isAbstain) tab5AbstainBanner.classList.remove('hidden');
      else tab5AbstainBanner.classList.add('hidden');
    }

    const probs = c.probabilities || {};
    const p1 = probs['Độ 1 (Superficial)'] || probs['Độ 1'] || 0.0;
    const p2 = probs['Độ 2 (Partial Thickness)'] || probs['Độ 2'] || 0.0;
    const p3 = probs['Độ 3 (Full Thickness)'] || probs['Độ 3'] || 0.0;

    const setProb = (idBar, idLabel, val) => {
      const barEl = document.getElementById(idBar);
      const labelEl = document.getElementById(idLabel);
      if (barEl) barEl.style.width = `${Math.min(100, val)}%`;
      if (labelEl) labelEl.innerText = `${val.toFixed(1).replace('.', ',')}%`;
    };

    setProb('bar-prob-1', 'label-prob-1', p1);
    setProb('bar-prob-2', 'label-prob-2', p2);
    setProb('bar-prob-3', 'label-prob-3', p3);

    const roiImg = document.getElementById('img-roi-display');
    if (roiImg) roiImg.src = imgs.roi || imgs.original || '';

    // 6. Tab 3 - Grad-CAM XAI
    const gradImg = document.getElementById('img-gradcam-display');
    if (gradImg) {
      gradImg.src = imgs.gradcam || imgs.overlay || imgs.original || '';
    }

    // 7. Tab 4 - Phác đồ bù dịch Parkland
    const totalMlEl = document.getElementById('parkland-total-ml');
    const ml8hEl = document.getElementById('parkland-8h-ml');
    const rate8hEl = document.getElementById('parkland-rate-8h');
    const ml16hEl = document.getElementById('parkland-16h-ml');
    const rate16hEl = document.getElementById('parkland-rate-16h');
    const subtextEl = document.getElementById('parkland-time-subtext');
    const elapsedPill = document.getElementById('parkland-elapsed-pill');
    const barElapsed = document.getElementById('bar-parkland-elapsed');
    const statusHeading = document.getElementById('parkland-status-heading');
    const fluidTypeLabel = document.getElementById('parkland-fluid-type-label');
    const indicationAlert = document.getElementById('parkland-indication-alert');
    const phaseContainer = document.getElementById('parkland-phase-container');

    const tbsa = m.estimated_tbsa_pct || 0.0;
    const isIndicated = (pk && typeof pk.indicated === 'boolean') ? pk.indicated : (tbsa >= 10.0);
    const calculatedMl = (pk && pk.calculated_raw_ml) || (4.0 * (p.weight_kg || 68.0) * tbsa);
    const normBanner = document.getElementById('parkland-normalization-banner');
    if (normBanner) {
      if (m.calibration_mode === 'ruler' && m.area_cm2) {
        normBanner.innerHTML = `
          <strong>Quy đổi Vật lý trước khi tính Parkland:</strong>
          Đo trực tiếp từ thước chuẩn trong ảnh được <code>${Number(m.area_cm2).toFixed(1).replace('.', ',')} cm²</code>
          chia cho tổng diện tích da cơ thể (BSA Mosteller) → <strong>${formatTBSA(tbsa)}% TBSA thực tế</strong>.
        `;
      } else {
        normBanner.innerHTML = `
          <strong>Chuẩn hóa Giải phẫu trước khi tính Parkland:</strong>
          Từ <code>${pctFrame.toFixed(1).replace('.', ',')}%</code> khung ảnh → tách phông nền còn
          <code>${pctBurnSkin.toFixed(1).replace('.', ',')}%</code> diện tích da vùng <strong>${regName} (${regWeight}%)</strong>
          ${regCov < 0.99 ? `× <code>${Math.round(regCov * 100)}%</code> khung vùng ` : ''}
          → <strong>${formatTBSA(tbsa)}% TBSA thực tế</strong> (tránh sai lệch dịch truyền do tiêu cự camera).
        `;
      }
    }

    if (indicationAlert) {
      indicationAlert.classList.remove('hidden');
      if (!isIndicated) {
        if (isDegree3) {
          indicationAlert.className = 'p-4 rounded-xl bg-amber-50 border border-amber-300 text-amber-950 flex flex-col gap-1.5';
          indicationAlert.innerHTML = `
            <div class="flex items-center gap-2 font-bold text-amber-900 text-[14px]">
              <span class="material-symbols-outlined text-[20px] text-amber-600">warning</span>
              <span>BỎNG ĐỘ 3 DIỆN TÍCH NHỎ (${formatTBSA(tbsa)}% TBSA) — KHÔNG CÓ NGUY CƠ SỐC BỎNG</span>
            </div>
            <p class="text-[13px] text-amber-900/90 leading-relaxed">
              • <strong>Không truyền dịch tĩnh mạch:</strong> Diện tích vết bỏng nhỏ (${formatTBSA(tbsa)}% TBSA) không làm thất thoát thể tích tuần hoàn. Tuyệt đối KHÔNG truyền dịch tĩnh mạch để tránh phù phổi cấp và suy tim.<br>
              • <strong>Chỉ định bắt buộc:</strong> Bù nước đường uống (Oresol) theo nhu cầu. Bắt buộc đến cơ sở y tế để bác sĩ ngoại khoa Bỏng <strong>CẮT LỌC MÔ HOẠI TỬ (Debridement)</strong> và tiêm phòng uốn ván.
            </p>
          `;
        } else {
          indicationAlert.className = 'p-4 rounded-xl bg-emerald-50 border border-emerald-300 text-emerald-950 flex flex-col gap-1.5';
          indicationAlert.innerHTML = `
            <div class="flex items-center gap-2 font-bold text-emerald-900 text-[14px]">
              <span class="material-symbols-outlined text-[20px] text-emerald-600">check_circle</span>
              <span>DIỆN TÍCH TỔN THƯƠNG NHỎ (${formatTBSA(tbsa)}% TBSA) — KHÔNG CẦN TRUYỀN DỊCH TĨNH MẠCH</span>
            </div>
            <p class="text-[13px] text-emerald-900/90 leading-relaxed">
              Vết bỏng chiếm ${formatTBSA(tbsa)}% TBSA, cơ thể không bị sốc mất nước. Bệnh nhân chỉ cần bù nước qua đường uống (Oresol, nước ấm) và tiến hành chăm sóc sát trùng, băng gạc tại chỗ theo hướng dẫn ở <strong>Tab 5</strong>.
            </p>
          `;
        }
      } else {
        indicationAlert.className = 'p-4 rounded-xl bg-red-50 border border-red-300 text-red-950 flex flex-col gap-1.5';
        indicationAlert.innerHTML = `
          <div class="flex items-center gap-2 font-bold text-red-900 text-[14px]">
            <span class="material-symbols-outlined text-[20px] text-red-600">emergency</span>
            <span>CHỈ ĐỊNH CẤP CỨU TRUYỀN DỊCH HỒI SỨC THEO PHÁC ĐỒ PARKLAND</span>
          </div>
          <p class="text-[13px] text-red-900/90 leading-relaxed">
            Diện tích tổn thương rộng (${formatTBSA(tbsa)}% TBSA), nguy cơ sốc giảm thể tích cao. Bắt buộc truyền 50% tổng lượng dịch trong 8 giờ đầu tính từ lúc bị bỏng.
          </p>
        `;
      }
    }

    if (statusHeading) {
      statusHeading.innerText = isIndicated ? 'Tổng lượng dịch truyền trong 24 giờ đầu' : 'Tình trạng: Không có chỉ định truyền dịch tĩnh mạch';
    }
    if (fluidTypeLabel) {
      fluidTypeLabel.innerText = isIndicated ? 'Dung dịch Ringer Lactate' : 'Bù nước đường uống (Oresol)';
    }

    if (totalMlEl) {
      if (isIndicated) {
        totalMlEl.className = 'text-[28px] font-extrabold text-[#007AFF] tabular-nums block';
        totalMlEl.innerText = `${Math.round(calculatedMl).toLocaleString()} ml`;
      } else {
        totalMlEl.className = 'text-[22px] font-extrabold text-emerald-600 tabular-nums block';
        totalMlEl.innerText = '0 ml (Uống Oresol)';
      }
    }

    if (phaseContainer) {
      if (!isIndicated) {
        phaseContainer.classList.add('opacity-40');
      } else {
        phaseContainer.classList.remove('opacity-40');
      }
    }

    const burnTime = (m.burn_time_hours !== undefined && m.burn_time_hours !== null) ? Number(m.burn_time_hours) : 2.0;
    const rem8h = Math.max(0.5, 8.0 - burnTime);
    const first8h = isIndicated ? ((pk && pk.first_8h_ml) || (calculatedMl * 0.5)) : 0;
    const rate8h = isIndicated ? ((pk && pk.rate_first_8h_ml_h) || (first8h / rem8h)) : 0;
    const next16h = isIndicated ? (calculatedMl - first8h) : 0;
    const rate16h = isIndicated ? ((pk && pk.rate_next_16h_ml_h) || (next16h / 16.0)) : 0;

    if (ml8hEl) ml8hEl.innerText = isIndicated ? `${Math.round(first8h).toLocaleString()} ml` : '0 ml';
    if (rate8hEl) rate8hEl.innerText = isIndicated ? `${rate8h.toFixed(1)} ml/h (~${Math.round(rate8h / 3)} gtt/p)` : 'Không truyền tĩnh mạch';
    if (ml16hEl) ml16hEl.innerText = isIndicated ? `${Math.round(next16h).toLocaleString()} ml` : '0 ml';
    if (rate16hEl) rate16hEl.innerText = isIndicated ? `${rate16h.toFixed(1)} ml/h (~${Math.round(rate16h / 3)} gtt/p)` : 'Không truyền tĩnh mạch';

    if (subtextEl) {
      subtextEl.innerText = isIndicated ?
        `Đã tiêu tốn ${burnTime}h từ khi bỏng · Còn lại ${rem8h.toFixed(1)}h để truyền hết 50% giai đoạn 1` :
        `Tổn thương diện tích nhỏ, không cần chia lộ trình truyền dịch tĩnh mạch.`;
    }
    if (elapsedPill) elapsedPill.innerText = `Đã qua: ${burnTime}h`;
    const elapsedPct = Math.min(100, (burnTime / 8.0) * 100);
    if (barElapsed) barElapsed.style.width = isIndicated ? `${elapsedPct}%` : '0%';

    // Mặc định chuyển về Tab 1
    this.switchResultTab('segmentation');
  }
}

