document.addEventListener("DOMContentLoaded", function () {
  document.getElementById("grabChips").addEventListener("click", () => {
    // Send message to content script to grab the chip text
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, { action: "grabChips" }, (response) => {


      if (response && response.data) {
        const chipList = document.getElementById("chipList");
        chipList.innerHTML = ''; // Clear the list

        response.data.forEach((text) => {
          let listItem = document.createElement('li');
          listItem.textContent = text;
          chipList.appendChild(listItem);
        });

      } else {
            console.error("No data received from content script."); // yep this gets triggered
      }

      });
    });
  });
});
