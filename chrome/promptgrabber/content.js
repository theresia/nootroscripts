/*
// the command to run on console

const links = document.querySelectorAll('span.mdc-evolution-chip__text-label');
links.forEach(link => console.log(`${link.textContent}`));

// the HTML

<mat-chip-listbox _ngcontent-ng-c2482895504="" aria-label="Follow up question chips" class="mdc-evolution-chip-set mat-mdc-chip-listbox mat-mdc-chip-set questions ng-star-inserted" role="listbox" tabindex="0" aria-required="false" aria-disabled="false" aria-multiselectable="false" aria-orientation="horizontal" style=""><div role="presentation" class="mdc-evolution-chip-set__chips"><mat-chip-option _ngcontent-ng-c2482895504="" appearance="hairline-suggestive" class="mat-mdc-chip mat-mdc-chip-option question mat-primary mdc-evolution-chip mat-mdc-standard-chip mdc-evolution-chip--filter mdc-evolution-chip--selectable mdc-evolution-chip--selecting mdc-evolution-chip--with-primary-graphic gmat-mdc-chip gmat-suggestive-chip gmat-hairline-chip ng-star-inserted" mat-ripple-loader-class-name="mat-mdc-chip-ripple" jslog="189028;track:generic_click,impression" id="mat-mdc-chip-26" role="presentation"><span class="mat-mdc-chip-focus-overlay"></span><span class="mdc-evolution-chip__cell mdc-evolution-chip__cell--primary"><button matchipaction="" role="option" class="mdc-evolution-chip__action mat-mdc-chip-action mdc-evolution-chip__action--primary" type="button" aria-selected="false" aria-label="Follow up question Button" aria-describedby="mat-mdc-chip-26-aria-description" tabindex="-1" aria-disabled="false"><span class="mdc-evolution-chip__graphic mat-mdc-chip-graphic ng-star-inserted"><span class="mdc-evolution-chip__checkmark"><svg viewBox="-2 -3 30 30" focusable="false" aria-hidden="true" class="mdc-evolution-chip__checkmark-svg"><path fill="none" stroke="currentColor" d="M1.73,12.91 8.1,19.28 22.79,4.59" class="mdc-evolution-chip__checkmark-path"></path></svg></span></span><!----><span class="mdc-evolution-chip__text-label mat-mdc-chip-action-label"> What specific examples does Hayakawa provide to illustrate the difference between intensional and extensional meanings? <span class="mat-mdc-chip-primary-focus-indicator mat-focus-indicator"></span></span></button></span><!----><span class="cdk-visually-hidden" id="mat-mdc-chip-26-aria-description"></span><span class="mat-ripple mat-mdc-chip-ripple"></span></mat-chip-option><!----><mat-chip-option _ngcontent-ng-c2482895504="" appearance="hairline-suggestive" class="mat-mdc-chip mat-mdc-chip-option question mat-primary mdc-evolution-chip mat-mdc-standard-chip mdc-evolution-chip--filter mdc-evolution-chip--selectable mdc-evolution-chip--selecting mdc-evolution-chip--with-primary-graphic gmat-mdc-chip gmat-suggestive-chip gmat-hairline-chip ng-star-inserted" mat-ripple-loader-class-name="mat-mdc-chip-ripple" jslog="189028;track:generic_click,impression" id="mat-mdc-chip-27" role="presentation"><span class="mat-mdc-chip-focus-overlay"></span><span class="mdc-evolution-chip__cell mdc-evolution-chip__cell--primary"><button matchipaction="" role="option" class="mdc-evolution-chip__action mat-mdc-chip-action mdc-evolution-chip__action--primary" type="button" aria-selected="false" aria-label="Follow up question Button" aria-describedby="mat-mdc-chip-27-aria-description" tabindex="-1" aria-disabled="false"><span class="mdc-evolution-chip__graphic mat-mdc-chip-graphic ng-star-inserted"><span class="mdc-evolution-chip__checkmark"><svg viewBox="-2 -3 30 30" focusable="false" aria-hidden="true" class="mdc-evolution-chip__checkmark-svg"><path fill="none" stroke="currentColor" d="M1.73,12.91 8.1,19.28 22.79,4.59" class="mdc-evolution-chip__checkmark-path"></path></svg></span></span><!----><span class="mdc-evolution-chip__text-label mat-mdc-chip-action-label"> How does Hayakawa contend that the two-valued orientation can be detrimental to effective communication and decision-making? <span class="mat-mdc-chip-primary-focus-indicator mat-focus-indicator"></span></span></button></span><!----><span class="cdk-visually-hidden" id="mat-mdc-chip-27-aria-description"></span><span class="mat-ripple mat-mdc-chip-ripple"></span></mat-chip-option><!----><mat-chip-option _ngcontent-ng-c2482895504="" appearance="hairline-suggestive" class="mat-mdc-chip mat-mdc-chip-option question mat-primary mdc-evolution-chip mat-mdc-standard-chip mdc-evolution-chip--filter mdc-evolution-chip--selectable mdc-evolution-chip--selecting mdc-evolution-chip--with-primary-graphic gmat-mdc-chip gmat-suggestive-chip gmat-hairline-chip ng-star-inserted" mat-ripple-loader-class-name="mat-mdc-chip-ripple" jslog="189028;track:generic_click,impression" id="mat-mdc-chip-28" role="presentation"><span class="mat-mdc-chip-focus-overlay"></span><span class="mdc-evolution-chip__cell mdc-evolution-chip__cell--primary"><button matchipaction="" role="option" class="mdc-evolution-chip__action mat-mdc-chip-action mdc-evolution-chip__action--primary" type="button" aria-selected="false" aria-label="Follow up question Button" aria-describedby="mat-mdc-chip-28-aria-description" tabindex="-1" aria-disabled="false"><span class="mdc-evolution-chip__graphic mat-mdc-chip-graphic ng-star-inserted"><span class="mdc-evolution-chip__checkmark"><svg viewBox="-2 -3 30 30" focusable="false" aria-hidden="true" class="mdc-evolution-chip__checkmark-svg"><path fill="none" stroke="currentColor" d="M1.73,12.91 8.1,19.28 22.79,4.59" class="mdc-evolution-chip__checkmark-path"></path></svg></span></span><!----><span class="mdc-evolution-chip__text-label mat-mdc-chip-action-label"> What are Hayakawa's recommendations for achieving an extensional orientation in language and thought? <span class="mat-mdc-chip-primary-focus-indicator mat-focus-indicator"></span></span></button></span><!----><span class="cdk-visually-hidden" id="mat-mdc-chip-28-aria-description"></span><span class="mat-ripple mat-mdc-chip-ripple"></span></mat-chip-option><!----><!----></div></mat-chip-listbox>
*/

// Function to grab the chip text
function grabChipText() {
  let chipTexts = [];
  
  // Find all elements with the class 'mdc-evolution-chip__text-label'
  const chips = document.querySelectorAll('.mdc-evolution-chip__text-label');
  const chipTexts1 = Array.from(chips).map(chip => chip.textContent.trim());
  console.log("Chip texts (1) found:", chipTexts1);
  
  // Loop through all found elements and get the inner text
  chips.forEach((chip) => {
    chipTexts.push(chip.innerText.trim());
  });
  
  // Return the collected texts
  console.log(chipTexts);
  return chipTexts;
}

// Send a message back to the background script or extension popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "grabChips") {
    const chipTexts = grabChipText();
    sendResponse({ data: chipTexts });
    return true;
  }
});
