// Content script - runs on every page
// This script helps collect additional page metadata

// Function to extract page metadata
function extractPageMetadata() {
    const getMetaContent = (name) => {
        const selectors = [
            `meta[name="${name}"]`,
            `meta[property="${name}"]`,
            `meta[property="og:${name}"]`,
            `meta[name="twitter:${name}"]`
        ];

        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
                return element.getAttribute('content') || '';
            }
        }
        return '';
    };

    // Extract various metadata
    const metadata = {
        title: document.title || '',
        description: getMetaContent('description') || getMetaContent('og:description') || '',
        keywords: getMetaContent('keywords') || '',
        author: getMetaContent('author') || '',
        ogTitle: getMetaContent('og:title') || '',
        ogDescription: getMetaContent('og:description') || '',
        ogImage: getMetaContent('og:image') || '',
        twitterTitle: getMetaContent('twitter:title') || '',
        twitterDescription: getMetaContent('twitter:description') || '',
        canonical: document.querySelector('link[rel="canonical"]')?.href || '',
        language: document.documentElement.lang || getMetaContent('language') || '',
        charset: document.charset || document.characterSet || ''
    };

    return metadata;
}

// Function to get structured data (JSON-LD)
function getStructuredData() {
    const scripts = document.querySelectorAll('script[type="application/ld+json"]');
    const structuredData = [];

    scripts.forEach(script => {
        try {
            const data = JSON.parse(script.textContent);
            structuredData.push(data);
        } catch (e) {
            // Invalid JSON, skip
        }
    });

    return structuredData;
}

// Function to get page readability info
function getReadabilityInfo() {
    const textContent = document.body.textContent || '';
    const wordCount = textContent.trim().split(/\s+/).filter(word => word.length > 0).length;
    const charCount = textContent.length;

    // Estimate reading time (average 200 words per minute)
    const readingTime = Math.ceil(wordCount / 200);

    return {
        wordCount,
        charCount,
        readingTime
    };
}

// Enhanced metadata collection function
function getEnhancedPageMetadata() {
    const basicMetadata = extractPageMetadata();
    const structuredData = getStructuredData();
    const readabilityInfo = getReadabilityInfo();

    return {
        ...basicMetadata,
        structuredData,
        readabilityInfo,
        url: window.location.href,
        timestamp: new Date().toISOString()
    };
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getPageMetadata') {
        const metadata = getEnhancedPageMetadata();
        sendResponse(metadata);
    }
});

// Auto-collect metadata when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Wait a bit for dynamic content to load
        setTimeout(() => {
            const metadata = getEnhancedPageMetadata();
            chrome.runtime.sendMessage({
                action: 'pageMetadata',
                metadata: metadata
            });
        }, 1000);
    });
} else {
    // Page already loaded
    setTimeout(() => {
        const metadata = getEnhancedPageMetadata();
        chrome.runtime.sendMessage({
            action: 'pageMetadata',
            metadata: metadata
        });
    }, 1000);
}