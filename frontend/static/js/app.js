/**
 * BurnScan - Master Application Controller
 * Điều phối toàn bộ luồng tiến trình 4 bước và khởi tạo sự kiện
 */

class AppWizard {
  static init() {
    this.setupDropzone();
    CaliperManager.init();
    HistoryManager.init();

    // Event listener cho phím Enter trong ô chat
    const chatInput = document.getElementById('chat-user-input');
    if (chatInput) {
      chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') ChatAssistant.send();
      });
    }
  }

  /**
   * Điều hướng giữa các bước 1 -> 4
   */
  static goToStep(step) {
    if (step === 2 && !appState.selectedFile && !appState.activeSessionData) {
      showToast('Vui lòng chọn ảnh trước khi sang bước 2', 'warning');
      return;
    }
    if (step === 4 && !appState.activeSessionData) {
      showToast('Chưa có kết quả phân tích. Vui lòng bấm Bắt đầu phân tích AI.', 'warning');
      return;
    }

    appState.currentStep = step;
    document.querySelectorAll('.step-view').forEach(v => v.classList.add('hidden'));
    const activeView = document.getElementById(`view-step-${step}`);
    if (activeView) activeView.classList.remove('hidden');

    // Cập nhật biểu tượng thanh điều hướng tiến trình
    for (let i = 1; i <= 4; i++) {
      const circle = document.getElementById(`circle-step-${i}`);
      const label = document.getElementById(`label-step-${i}`);
      const stepNav = document.getElementById(`step-nav-${i}`);

      if (i < step) {
        circle.className = 'w-7 h-7 rounded-full bg-emerald-600 text-white flex items-center justify-center text-[13px] font-semibold shadow-xs';
        circle.innerHTML = '<span class="material-symbols-outlined text-[15px]">check</span>';
        label.className = 'text-[14px] font-medium text-gray-800';
        stepNav?.classList.remove('opacity-50');
      } else if (i === step) {
        circle.className = 'w-7 h-7 rounded-full bg-[#007AFF] text-white flex items-center justify-center text-[13px] font-semibold shadow-xs';
        circle.innerText = i;
        label.className = 'text-[14px] font-semibold text-[#007AFF]';
        stepNav?.classList.remove('opacity-50');
      } else {
        circle.className = 'w-7 h-7 rounded-full bg-[rgba(120,120,128,0.15)] text-[rgba(60,60,67,0.7)] flex items-center justify-center text-[13px] font-medium';
        circle.innerText = i;
        label.className = 'text-[14px] font-medium text-[rgba(60,60,67,0.7)]';
        stepNav?.classList.add('opacity-50');
      }
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  /**
   * Thiết lập khu vực kéo thả Dropzone tải ảnh
   */
  static setupDropzone() {
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('file-input');
    if (!dropzone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('border-blue-500', 'bg-blue-50/20');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('border-blue-500', 'bg-blue-50/20');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        this.handleFileSelected(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        this.handleFileSelected(e.target.files[0]);
      }
    });
  }

  static cameraStream = null;

  static async triggerCamera() {
    const modal = document.getElementById('camera-modal');
    const video = document.getElementById('camera-video');
    const loading = document.getElementById('camera-loading');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast('Trình duyệt của bạn không hỗ trợ truy cập Webcam/Camera.', 'warning');
      return;
    }

    if (modal) modal.classList.remove('hidden');
    if (loading) loading.classList.remove('hidden');

    try {
      if (this.cameraStream) {
        this.cameraStream.getTracks().forEach(t => t.stop());
        this.cameraStream = null;
      }

      let stream = null;
      // Thử ưu tiên camera sau (trên điện thoại) hoặc webcam thường (trên máy tính)
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        });
      } catch (eConstrain) {
        console.warn('Fallback sang cấu hình video mặc định:', eConstrain);
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
      }

      this.cameraStream = stream;

      if (video) {
        video.muted = true;
        video.setAttribute('playsinline', 'true');
        video.setAttribute('muted', 'true');
        video.srcObject = stream;

        video.onloadedmetadata = () => {
          if (loading) loading.classList.add('hidden');
          video.play().catch(e => console.warn('Video play warning:', e));
        };
      }
    } catch (err) {
      console.error('Camera error:', err);
      if (modal) modal.classList.add('hidden');
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        showToast('Trình duyệt chưa được cấp quyền Camera. Hãy bấm vào biểu tượng Ổ khóa/Camera ở thanh địa chỉ để cấp quyền!', 'warning');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        showToast('Không tìm thấy thiết bị Webcam/Camera kết nối trên máy này.', 'warning');
      } else {
        showToast(`Không thể bật camera: ${err.message || err.name}`, 'warning');
      }
    }
  }

  static closeCamera() {
    const modal = document.getElementById('camera-modal');
    if (this.cameraStream) {
      this.cameraStream.getTracks().forEach(t => t.stop());
      this.cameraStream = null;
    }
    const video = document.getElementById('camera-video');
    if (video) video.srcObject = null;
    if (modal) modal.classList.add('hidden');
  }

  static capturePhoto() {
    const video = document.getElementById('camera-video');
    if (!video || !this.cameraStream) {
      showToast('Camera chưa sẵn sàng', 'warning');
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (!blob) {
        showToast('Không thể chụp ảnh từ camera', 'warning');
        return;
      }
      const filename = `camera_${Date.now()}.jpg`;
      const file = new File([blob], filename, { type: 'image/jpeg' });
      this.closeCamera();
      this.handleFileSelected(file);
      showToast('Đã chụp ảnh thành công!', 'check_circle');
    }, 'image/jpeg', 0.95);
  }

  static handleFileSelected(file) {
    this.dismissError();
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      this.showError('Định dạng không hỗ trợ', 'Vui lòng chọn ảnh định dạng JPG hoặc PNG.');
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      this.showError('Kích thước quá lớn', 'Vui lòng chọn ảnh có dung lượng dưới 15 MB.');
      return;
    }

    appState.selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      appState.selectedDataUrl = e.target.result;

      // Hiển thị Thumbnail Preview ở Bước 1
      const previewImg = document.getElementById('img-step-1-preview');
      const filenameEl = document.getElementById('preview-filename');
      const filesizeEl = document.getElementById('preview-filesize');
      const dropzone = document.getElementById('upload-dropzone');
      const previewCard = document.getElementById('upload-preview');
      const btnNext1 = document.getElementById('btn-next-step-1');

      if (previewImg) previewImg.src = appState.selectedDataUrl;
      if (filenameEl) filenameEl.innerText = file.name;
      if (filesizeEl) filesizeEl.innerText = `${(file.size / (1024 * 1024)).toFixed(2)} MB · Định dạng: ${file.type.split('/')[1].toUpperCase()}`;
      if (dropzone) dropzone.classList.add('hidden');
      if (previewCard) previewCard.classList.remove('hidden');
      if (btnNext1) btnNext1.removeAttribute('disabled');

      // Nạp ảnh lên Canvas Hiệu chuẩn ở Bước 2
      const calibImg = document.getElementById('img-calib-canvas');
      if (calibImg) calibImg.src = appState.selectedDataUrl;
    };
    reader.readAsDataURL(file);
  }

  static clearSelectedImage() {
    appState.selectedFile = null;
    appState.selectedDataUrl = null;
    const fileInput = document.getElementById('file-input');
    const dropzone = document.getElementById('upload-dropzone');
    const previewCard = document.getElementById('upload-preview');
    const btnNext1 = document.getElementById('btn-next-step-1');

    if (fileInput) fileInput.value = '';
    if (dropzone) dropzone.classList.remove('hidden');
    if (previewCard) previewCard.classList.add('hidden');
    if (btnNext1) btnNext1.setAttribute('disabled', 'true');
  }

  static showError(title, desc) {
    const banner = document.getElementById('error-banner');
    const titleEl = document.getElementById('error-banner-title');
    const descEl = document.getElementById('error-banner-desc');
    if (titleEl) titleEl.innerText = title;
    if (descEl) descEl.innerText = desc;
    if (banner) banner.classList.remove('hidden');
  }

  static dismissError() {
    const banner = document.getElementById('error-banner');
    if (banner) banner.classList.add('hidden');
  }

  /**
   * Chạy AI Pipeline phân tích (Chuyển sang Bước 3 rồi tới Bước 4)
   */
  static async startAiAnalysis() {
    if (!appState.selectedFile) {
      showToast('Vui lòng chọn ảnh trước', 'error');
      this.goToStep(1);
      return;
    }

    const weightInput = document.getElementById('input-patient-weight');
    const nameInput = document.getElementById('input-patient-name');
    const ageInput = document.getElementById('input-patient-age');
    const genderInput = document.getElementById('input-patient-gender');
    const burnTimeInput = document.getElementById('input-burn-time');

    const weight = weightInput ? parseFloat(weightInput.value) || 65.0 : 65.0;
    const patientName = nameInput ? (nameInput.value.trim() || 'Bệnh nhân vãng lai') : 'Bệnh nhân vãng lai';
    const patientAgeRaw = ageInput ? ageInput.value.trim() : '';
    const patientAge = patientAgeRaw ? parseInt(patientAgeRaw, 10) : null;
    const patientGender = genderInput ? genderInput.value || 'Nam' : 'Nam';
    const burnTime = burnTimeInput ? parseFloat(burnTimeInput.value) || 1.0 : 1.0;

    // Reset pipeline task icons before starting
    const t2 = document.getElementById('pipe-task-2')?.querySelector('.material-symbols-outlined');
    const t3 = document.getElementById('pipe-task-3')?.querySelector('.material-symbols-outlined');
    const t4 = document.getElementById('pipe-task-4')?.querySelector('.material-symbols-outlined');
    if (t2) { t2.className = 'material-symbols-outlined text-[20px] text-blue-600 animate-spin'; t2.innerText = 'progress_activity'; }
    if (t3) { t3.className = 'material-symbols-outlined text-[20px] text-gray-300'; t3.innerText = 'radio_button_unchecked'; }
    if (t4) { t4.className = 'material-symbols-outlined text-[20px] text-gray-300'; t4.innerText = 'radio_button_unchecked'; }

    // Chuyển sang Bước 3
    this.goToStep(3);

    const bar = document.getElementById('bar-analysis-progress');
    const label = document.getElementById('label-analysis-progress');

    let progress = 15;
    if (bar) bar.style.width = `${progress}%`;
    if (label) label.innerText = `${progress}%`;

    const timer = setInterval(() => {
      if (progress < 85) {
        progress += Math.floor(Math.random() * 15) + 5;
        if (progress > 85) progress = 85;
        if (bar) bar.style.width = `${progress}%`;
        if (label) label.innerText = `${progress}%`;

        if (progress > 30) {
          const t2Icon = document.getElementById('pipe-task-2')?.querySelector('.material-symbols-outlined');
          const t3Icon = document.getElementById('pipe-task-3')?.querySelector('.material-symbols-outlined');
          if (t2Icon) {
            t2Icon.className = 'material-symbols-outlined text-[20px] text-green-600';
            t2Icon.innerText = 'check_circle';
          }
          if (t3Icon) {
            t3Icon.className = 'material-symbols-outlined text-[20px] text-blue-600 animate-spin';
            t3Icon.innerText = 'progress_activity';
          }
        }
        if (progress > 60) {
          const t3Icon = document.getElementById('pipe-task-3')?.querySelector('.material-symbols-outlined');
          const t4Icon = document.getElementById('pipe-task-4')?.querySelector('.material-symbols-outlined');
          if (t3Icon) {
            t3Icon.className = 'material-symbols-outlined text-[20px] text-green-600';
            t3Icon.innerText = 'check_circle';
          }
          if (t4Icon) {
            t4Icon.className = 'material-symbols-outlined text-[20px] text-blue-600 animate-spin';
            t4Icon.innerText = 'progress_activity';
          }
        }
      }
    }, 350);

    const formData = new FormData();
    formData.append('file', appState.selectedFile);
    formData.append('pixels_per_cm', appState.calibPxPerCm);
    formData.append('body_part', appState.bodyPart);
    formData.append('weight_kg', weight);
    formData.append('burn_time_hours', burnTime);
    formData.append('patient_name', patientName);
    if (patientAge !== null && !Number.isNaN(patientAge)) {
      formData.append('patient_age', patientAge);
    }
    formData.append('patient_gender', patientGender);

    try {
      const data = await ApiService.analyzeBurn(formData);
      clearInterval(timer);

      if (bar) bar.style.width = '100%';
      if (label) label.innerText = '100%';

      appState.activeSessionData = data;

      setTimeout(() => {
        ResultsViewer.populateResults(data);
        this.goToStep(4);
        HistoryManager.loadSessions();
        showToast('Phân tích AI hoàn tất thành công!');
      }, 500);

    } catch (err) {
      clearInterval(timer);
      alert(`Lỗi phân tích: ${err.message}`);
      this.goToStep(2);
    }
  }

  /**
   * Xuất báo cáo PDF
   */
  static async exportPdf() {
    if (!appState.activeSessionData) {
      showToast('Chưa có dữ liệu ca khám để xuất PDF', 'error');
      return;
    }
    showToast('Đang tạo báo cáo bệnh án PDF...', 'hourglass_top');
    try {
      const data = await ApiService.exportPdf(appState.activeSessionData);
      if (data.pdf_url) {
        window.open(data.pdf_url, '_blank');
        showToast('Đã xuất báo cáo PDF thành công!');
      }
    } catch (err) {
      alert(`Lỗi xuất PDF: ${err.message}`);
    }
  }
}

// Khởi chạy khi DOM sẵn sàng
window.addEventListener('DOMContentLoaded', () => {
  AppWizard.init();
});

// Global UI bindings for modular HTML partials
window.goToStep = (step) => AppWizard.goToStep(step);
window.triggerCameraCapture = () => AppWizard.triggerCamera();
window.closeCameraModal = () => AppWizard.closeCamera();
window.captureCameraPhoto = () => AppWizard.capturePhoto();
window.clearSelectedImage = () => AppWizard.clearSelectedImage();
window.dismissError = () => AppWizard.dismissError();
window.startAiAnalysis = () => AppWizard.startAiAnalysis();
window.exportCurrentReportPdf = () => AppWizard.exportPdf();
window.switchMainView = (mode) => {
  if (mode === 'assessment') AppWizard.goToStep(appState.currentStep);
};

window.resetCaliper = () => CaliperManager.reset();
window.toggleCaliperVisibility = () => CaliperManager.toggleVisibility();
window.setCalibPreset = (val, name) => CaliperManager.setPreset(val, name);
window.recomputeRatio = () => CaliperManager.recomputeRatio();
window.onBodyPartChange = (val) => CaliperManager.setBodyPart(val);
window.setBodyPart = (val) => CaliperManager.setBodyPart(val);

window.onSplitSliderChange = (val) => ResultsViewer.onSplitSliderChange(val);
window.setViewerDisplayMode = (mode) => ResultsViewer.setDisplayMode(mode);
window.switchResultTab = (tab) => ResultsViewer.switchResultTab(tab);

window.toggleHistoryDrawer = (show) => HistoryManager.toggleDrawer(show);
window.debounceHistorySearch = (q) => HistoryManager.onSearchInput(q);
window.loadPastSession = (id) => HistoryManager.loadPastSession(id);
window.deleteSessionRecord = (id) => HistoryManager.deleteSession(id);

window.openAssistantModal = () => ChatAssistant.openModal();
window.closeAssistantModal = () => ChatAssistant.closeModal();
window.sendQuickChat = (text) => ChatAssistant.sendQuick(text);
window.sendChatMessage = () => ChatAssistant.send();

