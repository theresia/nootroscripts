document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.local.get('videoLinks', (result) => {
    const videoLinks = result.videoLinks || [];
    const videoLinksElement = document.getElementById('video-links');
    
    if (videoLinks.length === 0) {
      videoLinksElement.textContent = "EH NOOOO video links found.";
    } else {
      videoLinksElement.textContent = videoLinks.join('\n');
    }
  });
});