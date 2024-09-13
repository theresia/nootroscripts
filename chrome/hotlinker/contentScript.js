/*
// the HTML:

<div class="html5-video-container" data-layer="0"><video tabindex="-1" class="video-stream html5-main-video" webkit-playsinline="" playsinline="" controlslist="nodownload" src="https://chrt.fm/track/GB4AE5/www.buzzsprout.com/1783651/15708404-bjorn-lomborg-a-data-driven-approach-to-global-issues.mp3"></video></div>

----

// the brief / tutorial

https://chatgpt.com/c/66e28e48-9520-800e-b4a5-8cfffc329b29

### How the Components Work Together:

1. **Content Script (`contentScript.js`)**: This script runs on the target webpage, detects the video element using techniques like `MutationObserver`, and sends the video URL to the background script via `chrome.runtime.sendMessage`.

2. **Background Script (`background.js`)**: Listens for the message from the content script, stores the video URL in `chrome.storage.local`.

3. **Popup Script (`popup.js`)**: Retrieves the stored video URLs from `chrome.storage.local` and displays them in the popup.
*/

console.log('Content script running...');

console.log('Video elements found?');
const afs = document.querySelectorAll('video.video-stream');
console.log('Video elements found:', afs);
let videoLinks = [];
console.log('videoLinks 1:', videoLinks);

// Collect the video stream URLs
afs.forEach(af => {
  console.log('af.src:', af.src);
  videoLinks.push(af.src);
});
console.log('videoLinks 2:', videoLinks);

// Send the video links to the background script
chrome.runtime.sendMessage({ videoLinks: videoLinks }, (response) => {
  console.log("Video links sent to background script:", videoLinks);
});

/*
// Using MutationObserver to detect changes in the DOM and extract video links
const observer = new MutationObserver((mutations, observer) => {
  const afs = document.querySelectorAll('video.video-stream');
  console.log('Video elements found:', afs);
  if (afs.length > 0) {
    let videoLinks = [];
    
    afs.forEach(af => {
      videoLinks.push(af.src);
    });
    
    // Send the video links to the background script
    chrome.runtime.sendMessage({ videoLinks: videoLinks }, (response) => {
      console.log("Video links sent to background script:", videoLinks);
    });
    
    // Stop observing once the links are found
    observer.disconnect();
  }
});

// Observe the entire document for changes
observer.observe(document, {
  childList: true,
  subtree: true
});

*/

/*

document.addEventListener('DOMContentLoaded', () => {
  const afs = document.querySelectorAll('video.video-stream');
  console.log('Video elements found:', afs);
  let videoLinks = [];

  afs.forEach(af => {
    videoLinks.push(af.src);
  });

  // Send the video links to the background script
  chrome.runtime.sendMessage({ videoLinks: videoLinks }, (response) => {
    console.log("Video links sent to background script:", videoLinks);
  });
});

*/