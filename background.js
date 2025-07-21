let port  = chrome.runtime.connectNative("ping_pong");
const socket = new WebSocket("ws://0.0.0.0:8080/ws");

socket.addEventListener("open", () => {
  console.log("WebSocket connection established");
});

socket.onmessage = (event) => {
  console.log("Message from desktop app:", event.data);
  try {
    const message = JSON.parse(event.data);
    handleNativeMessage(message);
  } catch (error) {
    console.error("Error parsing message from desktop app:", error);
  }
};

// Listen for tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    collectTabData(tab);
  }
});

// Listen for tab activation - collect all tabs to ensure freshness
chrome.tabs.onActivated.addListener((activeInfo) => {
  collectAllTabs();
});

// Listen for new tabs
chrome.tabs.onCreated.addListener((tab) => {
  if (tab.url) {
    collectTabData(tab);
  }
});

// Listen for tab removal - remove from storage when tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  removeTabData(tabId);
});

// Listen for window focus changes
chrome.windows.onFocusChanged.addListener((windowId) => {
  if (windowId !== chrome.windows.WINDOW_ID_NONE) {
    collectAllTabs();
  }
});

// Periodic collection every 2 minutes to ensure data freshness
setInterval(() => {
  collectAllTabs();
}, 2 * 60 * 1000);

// Automatic sync to desktop app every 30 seconds
// setInterval(() => {
//   syncWithDesktopApp();
// }, 30 * 1000);

// Function to collect all tabs from all windows
async function collectAllTabs() {
  try {
    const windows = await chrome.windows.getAll({
      populate: true,
      windowTypes: ["normal"]
    });

    console.log(`Found ${windows.length} windows`);

    // Get all current tabs to sync with storage
    const allCurrentTabs = await chrome.tabs.query({});
    const currentTabIds = new Set(allCurrentTabs.map(tab => tab.id));

    for (const window of windows) {
      console.log(`Processing window ${window.id} with ${window.tabs.length} tabs`);

      for (const tab of window.tabs) {
        if (shouldCollectTab(tab)) {
          await collectTabData(tab, window.id);
        }
      }
    }

    // Clean up tabs that are no longer open
    await cleanupClosedTabs(currentTabIds);

  } catch (error) {
    console.error('Error collecting all tabs:', error);
  }
}

// Check if tab should be collected
function shouldCollectTab(tab) {
  if (!tab.url) return false;

  // Skip internal browser pages
  const skipProtocols = ['chrome://', 'chrome-extension://', 'moz-extension://', 'about:', 'edge://'];
  return !skipProtocols.some(protocol => tab.url.startsWith(protocol));
}

// Function to collect tab data
async function collectTabData(tab, windowId = null) {
  try {
    console.log(`Collecting data for tab: ${tab.title} (${tab.url})`);

    // Get window info if not provided
    if (!windowId && tab.windowId) {
      windowId = tab.windowId;
    }

    let windowInfo = null;
    if (windowId) {
      try {
        windowInfo = await chrome.windows.get(windowId);
      } catch (e) {
        console.warn('Could not get window info:', e);
      }
    }

    // Inject content script to get meta description
    let metadata = {};
    try {
      const results = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: getPageMetadata
      });
      metadata = results[0]?.result || {};
    } catch (error) {
      console.warn('Could not inject content script:', error);
    }

    const tabData = {
      id: tab.id,
      windowId: windowId,
      windowType: windowInfo?.type || 'normal',
      url: tab.url,
      title: tab.title || 'No title',
      description: metadata.description || 'No description available',
      favicon: tab.favIconUrl || '',
      lastAccessed: new Date().toISOString(),
      isActive: tab.active,
      isAudible: tab.audible,
      isMuted: tab.mutedInfo?.muted || false,
      isPinned: tab.pinned,
      keywords: metadata.keywords || '',
      author: metadata.author || '',
      ogTitle: metadata.ogTitle || '',
      ogDescription: metadata.ogDescription || '',
      ogImage: metadata.ogImage || '',
      canonical: metadata.canonical || '',
      language: metadata.language || '',
      readingTime: metadata.readabilityInfo?.readingTime || 0,
      wordCount: metadata.readabilityInfo?.wordCount || 0,
      status: tab.status
    };

    // Store tab data with improved deduplication
    await storeTabData(tabData);

  } catch (error) {
    console.error('Error collecting tab data:', error);
  }
}

// Function to get page metadata (injected into page)
function getPageMetadata() {
  const getMetaContent = (name) => {
    const selectors = [
      `meta[name="${name}"]`,
      `meta[property="${name}"]`,
      `meta[property="og:${name}"]`,
      `meta[name="twitter:${name}"]`,
      `meta[property="twitter:${name}"]`
    ];

    for (const selector of selectors) {
      const meta = document.querySelector(selector);
      if (meta) {
        return meta.getAttribute('content') || '';
      }
    }
    return '';
  };

  // Get reading time estimation
  const getReadabilityInfo = () => {
    const textContent = document.body.textContent || '';
    const wordCount = textContent.trim().split(/\s+/).filter(word => word.length > 0).length;
    const readingTime = Math.ceil(wordCount / 200); // 200 words per minute
    return { wordCount, readingTime };
  };

  return {
    description: getMetaContent('description') || getMetaContent('og:description') || getMetaContent('twitter:description'),
    keywords: getMetaContent('keywords'),
    author: getMetaContent('author'),
    ogTitle: getMetaContent('og:title'),
    ogDescription: getMetaContent('og:description'),
    ogImage: getMetaContent('og:image'),
    twitterTitle: getMetaContent('twitter:title'),
    twitterDescription: getMetaContent('twitter:description'),
    canonical: document.querySelector('link[rel="canonical"]')?.href || '',
    language: document.documentElement.lang || getMetaContent('language') || '',
    title: document.title,
    readabilityInfo: getReadabilityInfo()
  };
}

// Function to store tab data with improved deduplication
async function storeTabData(tabData) {
  try {
    // Get existing data
    const result = await chrome.storage.local.get(['openTabs', 'tabIndex']);
    let openTabs = result.openTabs || [];
    let tabIndex = result.tabIndex || {};

    // Use tab ID for deduplication (more reliable than URL+windowId)
    const tabId = tabData.id;

    // Check if tab already exists
    const existingIndex = tabIndex[tabId];

    if (existingIndex !== undefined && openTabs[existingIndex]) {
      // Update existing entry
      openTabs[existingIndex] = {
        ...openTabs[existingIndex],
        ...tabData,
        firstAccessed: openTabs[existingIndex].firstAccessed || tabData.lastAccessed,
        accessCount: (openTabs[existingIndex].accessCount || 0) + 1
      };
    } else {
      // Add new entry
      tabData.firstAccessed = tabData.lastAccessed;
      tabData.accessCount = 1;
      openTabs.push(tabData);
      tabIndex[tabId] = openTabs.length - 1;
    }

    // Keep only last 1000 entries to prevent storage overflow
    if (openTabs.length > 1000) {
      const removeCount = openTabs.length - 1000;
      openTabs.splice(0, removeCount);

      // Rebuild index after removal
      tabIndex = {};
      openTabs.forEach((tab, index) => {
        tabIndex[tab.id] = index;
      });
    }

    // Store updated data
    await chrome.storage.local.set({
      openTabs,
      tabIndex,
      lastUpdated: new Date().toISOString()
    });

    console.log(`Stored tab data: ${tabData.title} (Total open tabs: ${openTabs.length})`);

    // Auto-sync to desktop app
    syncWithDesktopApp();

  } catch (error) {
    console.error('Error storing tab data:', error);
  }
}

// Function to remove tab data when tab is closed
async function removeTabData(tabId) {
  try {
    const result = await chrome.storage.local.get(['openTabs', 'tabIndex']);
    let openTabs = result.openTabs || [];
    let tabIndex = result.tabIndex || {};

    // Remove the tab from storage
    const tabIndexToRemove = tabIndex[tabId];
    if (tabIndexToRemove !== undefined) {
      openTabs.splice(tabIndexToRemove, 1);

      // Rebuild index after removal
      tabIndex = {};
      openTabs.forEach((tab, index) => {
        tabIndex[tab.id] = index;
      });

      await chrome.storage.local.set({
        openTabs,
        tabIndex,
        lastUpdated: new Date().toISOString()
      });

      console.log(`Removed tab ${tabId} from storage`);
    }

  } catch (error) {
    console.error('Error removing tab data:', error);
  }
}

// Function to clean up tabs that are no longer open
async function cleanupClosedTabs(currentTabIds) {
  try {
    const result = await chrome.storage.local.get(['openTabs', 'tabIndex']);
    let openTabs = result.openTabs || [];
    let tabIndex = result.tabIndex || {};

    // Remove tabs that are no longer in current tabs
    const updatedTabs = openTabs.filter(tab => currentTabIds.has(tab.id));

    // Rebuild index
    tabIndex = {};
    updatedTabs.forEach((tab, index) => {
      tabIndex[tab.id] = index;
    });

    await chrome.storage.local.set({
      openTabs: updatedTabs,
      tabIndex,
      lastUpdated: new Date().toISOString()
    });

    console.log(`Cleaned up closed tabs. Open tabs: ${updatedTabs.length}`);

  } catch (error) {
    console.error('Error cleaning up closed tabs:', error);
  }
}

// Function to get stored tab data (for popup)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getTabData') {
    chrome.storage.local.get(['openTabs', 'lastUpdated'], (result) => {
      sendResponse({
        tabHistory: result.openTabs || [],
        lastUpdated: result.lastUpdated || null
      });
    });
    return true; // Will respond asynchronously
  }

  if (request.action === 'clearTabData') {
    chrome.storage.local.clear(() => {
      sendResponse({ success: true });
    });
    return true;
  }

  if (request.action === 'refreshAllTabs') {
    collectAllTabs().then(() => {
      sendResponse({ success: true });
    });
    return true;
  }

  if (request.action === 'getTabStats') {
    chrome.storage.local.get(['openTabs'], (result) => {
      const openTabs = result.openTabs || [];

      const domains = new Set();
      openTabs.forEach(tab => {
        try {
          domains.add(new URL(tab.url).hostname);
        } catch (e) {
          // Invalid URL, skip
        }
      });

      sendResponse({
        total: openTabs.length,
        open: openTabs.length,
        closed: 0, // No closed tabs stored
        domains: domains.size
      });
    });
    return true;
  }

  if (request.action === 'sendToDesktop') {
    chrome.storage.local.get(['openTabs', 'tabIndex', 'lastUpdated'], async (result) => {
      try {
        // const response = await fetch('http://localhost:8080/update-tabs', {
        //   method: 'POST',
        //   headers: { 'Content-Type': 'application/json' },
        //   body: JSON.stringify(result)
        // });

        const response = socket.send(JSON.stringify(result));
        console.log('Sent data to desktop app of length:', JSON.stringify(result).length);
      } catch (error) {
        sendResponse({ success: false, message: 'Desktop app not reachable' });
      }
    });
    return true; // Will respond asynchronously
  }

  if (request.action === 'openTab') {
    const { tab_id, window_id } = request;
    
    // Use promises instead of async/await in message listener
    const openTab = async () => {
      try {
        // First, activate the window if window_id is provided
        if (window_id) {
          await chrome.windows.update(window_id, { focused: true });
        }
        
        // Then activate the specific tab
        if (tab_id) {
          await chrome.tabs.update(tab_id, { active: true });
        }
        
        sendResponse({ success: true, message: 'Tab activated successfully' });
      } catch (error) {
        console.error('Error opening tab:', error);
        sendResponse({ success: false, message: 'Failed to open tab: ' + error.message });
      }
    };
    
    openTab();
    return true; // Will respond asynchronously
  }
});


// Send message to native host
function sendNativeMessage(message) {
  if (nativePort && nativeConnected) {
    try {
      nativePort.postMessage(message);
      console.log('Sent message to native host:', message);
    } catch (error) {
      console.error('Error sending message to native host:', error);
    }
  } else {
    console.warn('Native messaging not connected');
  }
}

// Handle open tab request from native app
async function handleOpenTabRequest(data) {
  const { tab_id, window_id } = data;
  
  try {
    // First, activate the window if window_id is provided
    if (window_id) {
      await chrome.windows.update(window_id, { focused: true });
    }
    
    // Then activate the specific tab
    if (tab_id) {
      await chrome.tabs.update(tab_id, { active: true });
    }
    
    console.log(`✅ Tab ${tab_id} activated successfully`);
    
    // Send confirmation back to native app
    sendNativeMessage({
      action: 'openTab',
      success: true,
      message: `Tab ${tab_id} activated successfully`
    });
    
  } catch (error) {
    console.error('Error opening tab:', error);
    sendNativeMessage({
      action: 'openTab',
      success: false,
      error: error.message
    });
  }
}


function handleNativeMessage(message) {
  const { action, data } = message;
  
  console.log(`Received ${action} from native-app:`, data);
  
  switch (action) {
    case 'openTab':
      // Native app wants to open a tab
      handleOpenTabRequest(data);
      break;
      
    // case 'searchTabs':
    //   // Native app wants search results
    //   handleSearchRequest(data);
    //   break;
      
    // case 'getTabStats':
    //   // Native app wants tab statistics
    //   handleStatsRequest(data);
    //   break;
      
    default:
      console.log('Unknown action from native host:', action);
  }
}


// port.onMessage.addListener((response) => {
//   console.log('Received message from native host:', response);
// })

port.onMessage.addListener((message) => {
  console.log('Received message from native host:', message);
  handleNativeMessage(message);
});

/*
Listen for the native messaging port closing.
*/
port.onDisconnect.addListener((port) => {
  if (port.error) {
    console.log(`Disconnected due to an error: ${port.error.message}`);
  } else {
    // The port closed for an unspecified reason. If this occurred right after
    // calling `browser.runtime.connectNative()` there may have been a problem
    // starting the the native messaging client in the first place.
    // https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Native_messaging#troubleshooting
    console.log(`Disconnected`, port);
  }
});

chrome.action.onClicked.addListener(() => {
  console.log("Sending:  ping");
  port.postMessage("ping");
});

async function syncWithDesktopApp() {
  try {
    const data = await chrome.storage.local.get(['openTabs', 'tabIndex', 'lastUpdated']);
    socket.send(JSON.stringify(data));
    // await fetch('http://localhost:8080/update-tabs', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(data)
    // });
  } catch (error) {
    console.log('Desktop app sync failed:', error);
  }
}