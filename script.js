const menuToggle = document.getElementById("menuToggle");
const optionsMenu = document.getElementById("optionsMenu");

menuToggle.addEventListener("click", () => {
  optionsMenu.style.display = optionsMenu.style.display === "block" ? "none" : "block";
});

// Optional: click anywhere else to close the menu
document.addEventListener("click", function(event) {
  if (!menuToggle.contains(event.target) && !optionsMenu.contains(event.target)) {
    optionsMenu.style.display = "none";
  }
});

