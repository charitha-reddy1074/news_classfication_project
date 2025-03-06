// to get current year
function getYear() {
    var currentDate = new Date();
    var currentYear = currentDate.getFullYear();
    document.querySelector("#displayYear").innerHTML = currentYear;
}

getYear();


// client section owl carousel
$(".client_owl-carousel").owlCarousel({
    loop: true,
    margin: 20,
    dots: false,
    nav: true,
    navText: [],
    autoplay: true,
    autoplayHoverPause: true,
    navText: [
        '<i class="fa fa-angle-left" aria-hidden="true"></i>',
        '<i class="fa fa-angle-right" aria-hidden="true"></i>'
    ],
    responsive: {
        0: {
            items: 1
        },
        600: {
            items: 2
        },
        1000: {
            items: 2
        }
    }
});

/* filepath: /c:/Users/chari/Downloads/finexo-html/js/custom.js */
function classifyNews() {
  const newsArticle = document.getElementById('news-article').value;
  const outputBox = document.getElementById('result');

  // Placeholder for classification logic
  const category = "example category"; // Replace with actual classification logic

  outputBox.innerHTML = `The given news belongs to <strong>${category}</strong> category`;
}