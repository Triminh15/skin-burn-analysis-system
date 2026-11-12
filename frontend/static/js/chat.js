/**
 * BurnScan - Medical RAG AI Chatbot Module
 * Trợ lý Y khoa RAG hỗ trợ bác sĩ chẩn đoán & xử trí vết bỏng
 */

class ChatAssistant {
  static escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  static openModal() {
    const modal = document.getElementById('assistant-modal');
    if (modal) modal.classList.remove('hidden');
  }

  static closeModal() {
    const modal = document.getElementById('assistant-modal');
    if (modal) modal.classList.add('hidden');
  }

  static sendQuick(text) {
    const input = document.getElementById('chat-user-input');
    if (input) {
      input.value = text;
      this.send();
    }
  }

  static async send() {
    const input = document.getElementById('chat-user-input');
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;
    input.value = '';

    const container = document.getElementById('chat-messages-container');
    if (!container) return;

    // 1. Render bong bóng tin nhắn của User
    const userBubble = document.createElement('div');
    userBubble.className = 'flex items-start justify-end gap-2.5';
    userBubble.innerHTML = `
      <div class="bg-blue-600 text-white p-3.5 rounded-2xl rounded-tr-sm text-[13px] shadow-xs max-w-[80%] leading-relaxed">
        ${this.escapeHtml(text)}
      </div>
    `;
    container.appendChild(userBubble);
    container.scrollTop = container.scrollHeight;

    // 2. Render bong bóng Loading
    const loadingBubble = document.createElement('div');
    loadingBubble.className = 'flex items-start gap-2.5';
    loadingBubble.innerHTML = `
      <div class="w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-[12px] shrink-0 mt-1">
        <span class="material-symbols-outlined text-[15px] animate-spin">progress_activity</span>
      </div>
      <div class="bg-white p-3 rounded-2xl rounded-tl-sm border border-gray-200 text-[13px] text-gray-500 shadow-xs">
        Đang tra cứu cơ sở tri thức y khoa Bộ Y tế...
      </div>
    `;
    container.appendChild(loadingBubble);
    container.scrollTop = container.scrollHeight;

    try {
      const data = await ApiService.sendChatMessage(text, appState.activeSessionData);
      container.removeChild(loadingBubble);

      // 3. Render phản hồi của Bác sĩ AI
      const botBubble = document.createElement('div');
      botBubble.className = 'flex items-start gap-2.5';
      botBubble.innerHTML = `
        <div class="w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-[12px] shrink-0 mt-1">
          <span class="material-symbols-outlined text-[15px]">smart_toy</span>
        </div>
        <div class="bg-white p-3.5 rounded-2xl rounded-tl-sm border border-gray-200 text-[13px] text-gray-800 shadow-xs max-w-[85%] leading-relaxed whitespace-pre-line">
          ${this.escapeHtml(data.reply)}
        </div>
      `;
      container.appendChild(botBubble);
      container.scrollTop = container.scrollHeight;

    } catch (err) {
      if (container.contains(loadingBubble)) container.removeChild(loadingBubble);
      alert(`Lỗi trợ lý y khoa: ${err.message}`);
    }
  }
}

