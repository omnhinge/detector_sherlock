// Background service worker for Detector Sherlock
chrome.runtime.onInstalled.addListener(() => {
  // Add right-click context menu item
  chrome.contextMenus.create({
    id: "sherlock-verify-selection",
    title: "Sherlock: Verify with Evidence",
    contexts: ["selection"]
  });
});

// When user clicks the right-click menu item
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "sherlock-verify-selection" && info.selectionText) {
    // Save selected text to extension storage
    chrome.storage.local.set({ pendingVerification: info.selectionText }, () => {
      // Open the extension popup action
      chrome.action.openPopup();
    });
  }
});
