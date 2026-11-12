/**
 * BurnScan - Backend API Service Layer
 * Giao tiếp HTTP với Backend FastAPI
 */

class ApiService {
  /**
   * Gửi ảnh và dữ liệu lâm sàng lên Backend để chạy AI phân tích
   */
  static async analyzeBurn(formData) {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || 'Không thể thực hiện phân tích vết bỏng');
    }
    return await response.json();
  }

  /**
   * Lấy danh sách lịch sử các ca khám từ CSDL SQLite
   */
  static async fetchSessions(searchQuery = '') {
    let url = '/api/sessions';
    if (searchQuery && searchQuery.trim()) {
      url += `?q=${encodeURIComponent(searchQuery.trim())}`;
    }
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Không thể tải lịch sử ca khám từ CSDL');
    }
    return await response.json();
  }

  /**
   * Lấy chi tiết toàn bộ một ca khám theo UUID
   */
  static async fetchSessionDetail(sessionId) {
    const response = await fetch(`/api/sessions/${sessionId}`);
    if (!response.ok) {
      throw new Error('Không tìm thấy dữ liệu ca khám này');
    }
    return await response.json();
  }

  /**
   * Xóa một ca khám khỏi CSDL SQLite
   */
  static async deleteSession(sessionId) {
    const response = await fetch(`/api/sessions/${sessionId}`, {
      method: 'DELETE'
    });
    if (!response.ok) {
      throw new Error('Không thể xóa ca khám này');
    }
    return await response.json();
  }

  /**
   * Xuất báo cáo bệnh án y khoa PDF
   */
  static async exportPdf(sessionData) {
    const response = await fetch('/api/export-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sessionData)
    });
    if (!response.ok) {
      throw new Error('Không thể tạo file báo cáo PDF');
    }
    return await response.json();
  }

  /**
   * Gửi câu hỏi lâm sàng tới Trợ lý Y khoa RAG
   */
  static async sendChatMessage(query, clinicalContext = null) {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        clinical_context: clinicalContext
      })
    });
    if (!response.ok) {
      throw new Error('Không thể kết nối tới Trợ lý Y khoa');
    }
    return await response.json();
  }
}

