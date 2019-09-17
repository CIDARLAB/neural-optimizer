/* Open the sidenav */
function openNav() {
  document.getElementById("mySidenav").style.width = "300px";
  //document.getElementById("main").style.marginLeft = "250px";
}

/* Close/hide the sidenav */
function closeNav() {
  document.getElementById("mySidenav").style.width = "0";
  //document.getElementById("main").style.marginLeft = "0";
}

function changeShape(x) {
  x.classList.toggle("change");
}