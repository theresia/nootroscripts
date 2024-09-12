chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.videoLinks) {
    console.log("Received video links:", request.videoLinks);  // Debugging log
    // Check if chrome.storage.local is defined
    if (chrome.storage && chrome.storage.local) {
      chrome.storage.local.set({ videoLinks: request.videoLinks }, () => {
        if (chrome.runtime.lastError) {
          console.error('Error setting storage:', chrome.runtime.lastError);
        } else {
          console.log('Video links stored successfully');
        }
      });
    } else {
      console.error('chrome.storage.local is undefined');
    }
  }
});
