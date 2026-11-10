const conceptDemo = document.querySelector(".concept-demo");
const conceptButtons = [...document.querySelectorAll(".concept-switch button")];
const conceptPlay = document.querySelector(".concept-play");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let conceptPaused = false;
let conceptVisible = false;
let conceptTimer = null;
function updateConceptAnimation() {
  window.clearInterval(conceptTimer);
  conceptTimer = null;
  if (conceptPaused || reducedMotion.matches || !conceptVisible || document.visibilityState !== "visible") return;
  conceptTimer = window.setInterval(() => {
    conceptDemo.dataset.phase = String((Number(conceptDemo.dataset.phase) + 1) % 5);
  }, 1500);
}
conceptButtons.forEach((button) => {
  button.addEventListener("click", () => {
    conceptDemo.dataset.mode = button.dataset.concept;
    conceptDemo.dataset.phase = "0";
    conceptButtons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    updateConceptAnimation();
  });
});
conceptPlay.addEventListener("click", () => {
  conceptPaused = !conceptPaused;
  conceptPlay.setAttribute("aria-pressed", String(conceptPaused));
  conceptPlay.setAttribute("aria-label", conceptPaused ? "Play diagram animation" : "Pause diagram animation");
  conceptPlay.title = conceptPaused ? "Play animation" : "Pause animation";
  conceptPlay.textContent = conceptPaused ? "▶" : "Ⅱ";
  updateConceptAnimation();
});
new IntersectionObserver(([entry]) => {
  conceptVisible = entry.isIntersecting;
  updateConceptAnimation();
}).observe(conceptDemo);
reducedMotion.addEventListener("change", updateConceptAnimation);
document.addEventListener("visibilitychange", updateConceptAnimation);


document.querySelectorAll('[role="tablist"]').forEach((tabList) => {
  const tabs = [...tabList.querySelectorAll('[role="tab"]')];
  function selectTab(tab, moveFocus = false) {
    tabs.forEach((item) => {
      const selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
      document.getElementById(item.getAttribute("aria-controls")).hidden = !selected;
    });
    if (moveFocus) tab.focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
        : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      selectTab(tabs[next], true);
    });
  });
});

// Load the page counter only on the public site, keeping local previews out.
if (window.location.origin === "https://madrid1011.github.io"
    && window.location.pathname === "/SCARF/") {
  const counterScript = document.createElement("script");
  counterScript.src = "https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js";
  counterScript.async = true;
  document.head.appendChild(counterScript);
}
