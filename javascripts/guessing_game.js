var finished = false;
var guesses = 0;
var colors = ["blue", "cyan", "gold", "gray", "green", "magenta", "orange", "red", "white", "yellow"];
var target;

// Main function of this file.
function doGame() {
    // Draw a dope circle.
    drawCircle();


    var rand_num = Math.floor(Math.random() * colors.length);
    target = colors[rand_num];

    while (!finished) {
        var guessInputText = prompt("I am thinking of one of these colors:\n\n"
                                + "blue, cyan, gold, gray, green, magenta, "
                                + "orange, red, white, yellow\n\n"
                                + "What color am I thinking of?");
        var guessInput = guessInputText;
        guesses += 1;
        finished = checkGuess(guessInput);
    }
}

function checkGuess(guessInput) {
    // If color not found in COLORS array.
    if (colors.indexOf(guessInput) < 0) {
        alert("Sorry, I don't recognize your color.\n\n"
            + "Please try again.");
            return false;
    } else if (guessInput > target) {
        alert("Sorry, your guess is not correct!\n\n"
            + "Hint: Your color is alphabetically higher than mine.\n\n"
            + "Please try again.");
        return false;
    } else if (guessInput < target) {
        alert("Sorry, your guess is not correct!\n\n"
            + "Hint: Your color is alphabetically lower than mine.\n\n"
            + "Please try again.");
        return false;
    } else if (guessInput == target) {
        myBody=document.getElementsByTagName("body")[0];
        myBody.style.background=target;
        alert("Congrats, you got the right color.\n\n"
            + "It took you " + guesses + " guesses to finish.");
        return true;
    } else {
        return true;
    }
}


function drawCircle() {
    var c = document.getElementById("myCanvas");
    var ctx = c.getContext("2d");
    ctx.beginPath();
    ctx.arc(95, 50, 40, 0, 2 * Math.PI);
    ctx.stroke();
}
