// Popup script for displaying tab data
let allTabData = [];
let filteredTabData = [];
let lastUpdated = null;

// DOM elements
const tabList = document.getElementById('tabList');
const searchBox = document.getElementById('searchBox');
const refreshBtn = document.getElementById('refreshBtn');
const exportBtn = document.getElementById('exportBtn');
const clearBtn = document.getElementById('clearBtn');
const totalTabs = document.getElementById('totalTabs');
const uniqueDomains = document.getElementById('uniqueDomains');

// Initialize popup
document.addEventListener('DOMContentLoaded', () => {
  loadTabData();
  setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
  refreshBtn.addEventListener('click', () => {
    refreshBtn.textContent = 'Refreshing...';
    refreshBtn.disabled = true;
    
    chrome.runtime.sendMessage({ action: 'refreshAllTabs' }, (response) => {
      if (response.success) {
        loadTabData();
      }
      refreshBtn.textContent = 'Refresh';
      refreshBtn.disabled = false;
    });
  });
  
  exportBtn.addEventListener('click', exportData);
  
  // Add Send to Desktop button functionality
  const sendToDesktopBtn = document.getElementById('sendToDesktopBtn');
  sendToDesktopBtn.addEventListener('click', () => {
    sendToDesktopBtn.textContent = 'Sending...';
    sendToDesktopBtn.disabled = true;
    
    chrome.runtime.sendMessage({ action: 'sendToDesktop' }, (response) => {
      if (response.success) {
        alert('Data sent to desktop app successfully!');
      } else {
        alert('Failed to send data: ' + response.message);
      }
      sendToDesktopBtn.textContent = 'Send to Desktop';
      sendToDesktopBtn.disabled = false;
    });
  });
  
  clearBtn.addEventListener('click', clearAllData);
  searchBox.addEventListener('input', handleSearch);
}

// Load tab data from storage
function loadTabData() {
  chrome.runtime.sendMessage({ action: 'getTabData' }, (response) => {
    if (response && response.tabHistory) {
      allTabData = response.tabHistory;
      lastUpdated = response.lastUpdated;
      filteredTabData = [...allTabData];
      updateStats();
      renderTabList();
    }
  });
}

// Update statistics
function updateStats() {
  chrome.runtime.sendMessage({ action: 'getTabStats' }, (response) => {
    if (response) {
      totalTabs.textContent = response.total;
      uniqueDomains.textContent = response.domains;
      
      // Add additional stats
      const statsDiv = document.getElementById('stats');
      
      // Remove existing additional stats
      const existingStats = statsDiv.querySelectorAll('.additional-stat');
      existingStats.forEach(stat => stat.remove());
      
      // Add new stats
      const additionalStats = [
        { label: 'Open Tabs:', value: response.open }
      ];
      
      additionalStats.forEach(stat => {
        const statDiv = document.createElement('div');
        statDiv.className = 'stat-item additional-stat';
        statDiv.innerHTML = `
          <span class="stat-label">${stat.label}</span>
          <span class="stat-value">${stat.value}</span>
        `;
        statsDiv.appendChild(statDiv);
      });
      
      // Add last updated info
      if (lastUpdated) {
        const lastUpdateDiv = document.createElement('div');
        lastUpdateDiv.className = 'stat-item additional-stat';
        lastUpdateDiv.innerHTML = `
          <span class="stat-label">Last Updated:</span>
          <span class="stat-value" style="font-size: 10px;">${new Date(lastUpdated).toLocaleString()}</span>
        `;
        statsDiv.appendChild(lastUpdateDiv);
      }
    }
  });
}

// Render tab list
function renderTabList() {
  if (filteredTabData.length === 0) {
    tabList.innerHTML = `
      <div class="empty-state">
        <p>No open tabs found.</p>
        ${allTabData.length === 0 ? '<p>Browse some websites to start collecting data!</p>' : '<p>Try adjusting your search.</p>'}
      </div>
    `;
    return;
  }

  // Sort by last accessed (most recent first)
  filteredTabData.sort((a, b) => new Date(b.lastAccessed) - new Date(a.lastAccessed));

  const tabItems = filteredTabData.map(tab => {
    const favicon = tab.favicon ? 
      `<img src="${tab.favicon}" class="tab-favicon" alt="favicon">` : 
      '<div class="tab-favicon" style="background-color: #6c757d;"></div>';
    
    const description = tab.description && tab.description !== 'No description available' ? 
      `<div class="tab-description">${escapeHtml(tab.description)}</div>` : 
      '<div class="tab-description" style="font-style: italic; color: #adb5bd;">No description available</div>';
    
    const lastAccessed = new Date(tab.lastAccessed).toLocaleString();
    const firstAccessed = tab.firstAccessed ? new Date(tab.firstAccessed).toLocaleString() : 'Unknown';
    
    // Status indicators
    const statusIndicators = [];
    if (tab.isActive) statusIndicators.push('<span class="status-badge active">Active</span>');
    if (tab.isPinned) statusIndicators.push('<span class="status-badge pinned">Pinned</span>');
    if (tab.isAudible) statusIndicators.push('<span class="status-badge audible">Audio</span>');
    if (tab.isMuted) statusIndicators.push('<span class="status-badge muted">Muted</span>');
    
    const statusHtml = statusIndicators.length > 0 ? 
      `<div class="status-indicators">${statusIndicators.join('')}</div>` : '';
    
    // Additional info
    const additionalInfo = [];
    if (tab.accessCount > 1) additionalInfo.push(`Accessed ${tab.accessCount} times`);
    if (tab.readingTime > 0) additionalInfo.push(`${tab.readingTime} min read`);
    if (tab.wordCount > 0) additionalInfo.push(`${tab.wordCount} words`);
    if (tab.language) additionalInfo.push(`Language: ${tab.language}`);
    
    const additionalInfoHtml = additionalInfo.length > 0 ? 
      `<div class="additional-info">${additionalInfo.join(' • ')}</div>` : '';
    
    return `
      <div class="tab-item">
        <div class="tab-title">
          ${favicon}
          <span>${escapeHtml(tab.title)}</span>
        </div>
        <div class="tab-url">${escapeHtml(tab.url)}</div>
        ${statusHtml}
        ${description}
        ${additionalInfoHtml}
        <div class="tab-timestamp">
          Last accessed: ${lastAccessed}
          ${tab.firstAccessed !== tab.lastAccessed ? ` • First accessed: ${firstAccessed}` : ''}
        </div>
      </div>
    `;
  }).join('');

  tabList.innerHTML = tabItems;
}

// Handle search
function handleSearch(event) {
  const query = event.target.value.toLowerCase();
  
  if (!query) {
    filteredTabData = [...allTabData];
  } else {
    filteredTabData = allTabData.filter(tab => 
      tab.title.toLowerCase().includes(query) ||
      tab.url.toLowerCase().includes(query) ||
      tab.description.toLowerCase().includes(query)
    );
  }
  
  renderTabList();
}

// Export data
function exportData() {
  if (allTabData.length === 0) {
    alert('No data to export!');
    return;
  }

  const dataStr = JSON.stringify(allTabData, null, 2);
  const dataBlob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(dataBlob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = `open-tabs-${new Date().toISOString().split('T')[0]}.json`;
  link.click();
  
  URL.revokeObjectURL(url);
}

// Clear all data
function clearAllData() {
  if (confirm('Are you sure you want to clear all tab data? This cannot be undone.')) {
    chrome.runtime.sendMessage({ action: 'clearTabData' }, (response) => {
      if (response.success) {
        allTabData = [];
        filteredTabData = [];
        updateStats();
        renderTabList();
        alert('All tab data cleared successfully!');
      }
    });
  }
}

// Utility function to escape HTML
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}