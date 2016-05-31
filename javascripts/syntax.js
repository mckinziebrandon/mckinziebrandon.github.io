function getName() {
    return parseInt(prompt("What is your age?"));
}

/* Tell user that they are super cool. */
function greetUser() {
    console.log("About to tell user that they're super cool . . . ");
    alert("Breaking News: " + getName() + " is super cool.");
}

var pets = ["dog", "cat", "rabbit"];
alert(pets.join(" and "));

/* =========================================================== */
/* =========================================================== */

// Find path to a node
function handleClick(event) {
    // Make sure event is only executed one time.
    event.stopPropagation();
    // how...
    var node = event.target;
    // the fuck do we know ...
    var thisPath = node.nodeName;
    while (node.parentNode) {
        node = node.parentNode;
        thisPath = node.nodeName + " > " + thisPath;
    }
    alert(thisPath);
}

// nodes == HTML/SVG elements.
// Register click event handler for all nodes.
function attachHandler(node) {
    // Null nodes can't have a handler.
    if (node == null) return;
    // Tell node what to call when clicked.
    node.onclick = handleClick;
    // Do the same for all children nodes.
    for (var i = 0; i < node.childNodes.length; i++) {
        attachHandler(node.childNodes[i]);
    }
}

/* =========================================================== */
/* =========================================================== */
// ________ Locating Nodes in the DOM. ________

// This is what 'id' attribute is for in HTML.
// Use getElementById()
function change_color() {
    // getElementId
    document.getElementById("some id").style.color = "blue";
    // setAttribute
    the_node = getElementId("thisNode");
    the_node.setAttribute("style", "color:red");
}

/* =========================================================== */
/* =========================================================== */
// ________ Creating/Adding nodes. _______

// createElement()
result = document.createElement("div");
// Special commands for specific node.
result = document.createTextNode("Hello");

// Adding a hr to the DOM.
newItem = document.createelement("hr");
destParent = document.getElementByTagName("body")[0];
destParent.insertBefore(newItem, destParent.firstChild);
