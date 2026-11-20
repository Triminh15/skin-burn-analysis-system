/**
 * BurnScan - Patient Medical Records Drawer Module
 * Quản lý lịch sử bệnh án từ Cơ sở dữ liệu SQLite
 */

class HistoryManager {
  static init() {
    this.searchTimer = null;
    this.loadSessions();
  }

  /**
   * Bật / Tắt thanh trượt Drawer lịch sử bệnh án
   */
  static toggleDrawer(show) {
    const drawer = document.getElementById('history-drawer-backdrop');
    if (!drawer) return;

    if (show) {
      drawer.classList.remove('hidden');
      this.loadSessions();
    } else {
      drawer.classList.add('hidden');
    }
  }

  /**
   * Lắng nghe tìm kiếm bệnh nhân có Debounce 300ms
   */
  static onSearchInput(query) {
    clearTimeout(this.searchTimer);
    this.searchTimer = setTimeout(() => {
      this.loadSessions(query);
    }, 300);
  }

  /**
   * Nạp danh sách ca khám từ SQLite Backend
   */
  static async loadSessions(searchQuery = '') {
    const container = document.getElementById('history-items-container');
    const badgeCount = document.getElementById('history-badge-count');
    if (!container) return;

    try {
      const data = await ApiService.fetchSessions(searchQuery);
      const history = data.history || [];

      if (badgeCount) badgeCount.innerText = history.length;

      if (history.length === 0) {
        container.innerHTML = `
          <div class="text-center py-12 text-gray-400">
            <span class="material-symbols-outlined text-[36px]">folder_off</span>
            <p class="text-[13px] mt-2">Chưa tìm thấy ca bệnh án nào.</p>
          </div>
        `;
        return;
      }

      container.innerHTML = history.map(item => {
        let degreeColor = 'text-green-600';
        if (item.predicted_degree.includes('Độ 2')) degreeColor = 'text-orange-600';
        if (item.predicted_degree.includes('Độ 3')) degreeColor = 'text-red-600';

        return `
          <div class="p-3.5 rounded-2xl bg-white border border-gray-200/80 shadow-xs hover:border-blue-400 hover:shadow-md transition-all flex items-center justify-between gap-3 group">
            <div class="flex items-center gap-3 min-w-0">
              <div class="w-12 h-12 rounded-xl bg-gray-900 overflow-hidden shrink-0 border border-gray-200">
                <img src="${item.thumbnail || '/static/placeholder.jpg'}" alt="Thumb" class="w-full h-full object-cover">
              </div>
              <div class="min-w-0">
                <div class="flex items-center gap-2">
                  <span class="text-[14px] font-bold text-gray-900 truncate">${item.patient_name}</span>
                  <span class="text-[11px] font-mono px-1.5 py-0.2 rounded bg-gray-100 text-gray-600">${item.patient_code}</span>
                </div>
                <div class="flex items-center gap-2 text-[12px] text-gray-500 mt-0.5">
                  <span class="font-semibold ${degreeColor}">${item.predicted_degree}</span>
                  <span>·</span>
                  <span>TBSA: <strong>${item.estimated_tbsa_pct}%</strong></span>
                  <span>·</span>
                  <span>${item.created_at}</span>
                </div>
              </div>
            </div>

            <div class="flex items-center gap-1.5 shrink-0">
              <button onclick="HistoryManager.loadPastSession('${item.session_id}')" class="px-3 py-1.5 rounded-lg bg-blue-50 hover:bg-blue-600 hover:text-white text-blue-600 text-[12px] font-semibold transition-all">
                Xem lại
              </button>
              <button onclick="HistoryManager.deleteSession('${item.session_id}')" class="p-1.5 text-gray-400 hover:text-red-600 rounded-lg hover:bg-red-50 transition-all" title="Xóa ca này">
                <span class="material-symbols-outlined text-[16px]">delete</span>
              </button>
            </div>
          </div>
        `;
      }).join('');

    } catch (err) {
      container.innerHTML = `<p class="text-red-500 text-[13px] text-center py-6">Lỗi tải dữ liệu: ${err.message}</p>`;
    }
  }

  /**
   * Tải chi tiết ca khám cũ và nạp thẳng lên Bước 4 (không cần phân tích lại)
   */
  static async loadPastSession(sessionId) {
    try {
      const data = await ApiService.fetchSessionDetail(sessionId);
      appState.activeSessionData = data;
      this.toggleDrawer(false);

      ResultsViewer.populateResults(data);
      AppWizard.goToStep(4);

      showToast(`Đã nạp hồ sơ bệnh nhân: ${data.patient.name}`);
    } catch (err) {
      alert(`Lỗi nạp ca khám: ${err.message}`);
    }
  }

  /**
   * Xóa một ca khám khỏi CSDL SQLite
   */
  static async deleteSession(sessionId) {
    if (!confirm('Bạn có chắc chắn muốn xóa ca bệnh này khỏi cơ sở dữ liệu?')) return;
    try {
      await ApiService.deleteSession(sessionId);
      showToast('Đã xóa hồ sơ bệnh án thành công');
      this.loadSessions();
    } catch (err) {
      alert(`Lỗi khi xóa: ${err.message}`);
    }
  }
}

