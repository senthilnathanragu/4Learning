# Complete JavaScript Course Notes

---

## SECTION 1: GETTING STARTED

### What is JavaScript?
JavaScript is a high-level, interpreted, dynamic programming language primarily used to make web pages interactive. It runs in browsers and also on servers via Node.js.

- Originally created by Brendan Eich in 1995
- Follows the ECMAScript standard
- Can manipulate HTML/CSS, handle events, communicate with servers

### Setting Up the Development Environment
- Install **VS Code** as the code editor
- Install **Node.js** from nodejs.org (includes npm)
- Use browser DevTools (F12) to run quick JS snippets in the Console

### JavaScript in Browsers
```html
<!-- Inline in HTML -->
<script>
  console.log("Hello from browser");
</script>

<!-- External file (recommended) -->
<script src="app.js"></script>
```
Place `<script>` at the bottom of `<body>` so HTML loads first.

### Separation of Concerns
Keep HTML, CSS, and JavaScript in **separate files**. Don't mix behavior (JS) with structure (HTML).

### JavaScript in Node
Node.js lets you run JS outside the browser (on a server/terminal).
```bash
node app.js        # run a file
node               # open interactive REPL
```

---

## SECTION 2: BASICS

### Variables
A variable is a named container for storing data. Declared with `let`.
```js
let name = "John";
let age = 25;
let isActive = true;

name = "Jane"; // can be reassigned
```

### Constants
Declared with `const`. Value **cannot** be reassigned after declaration.
```js
const PI = 3.14159;
const MAX_SIZE = 100;

// PI = 3; // ERROR: Assignment to constant variable
```

### Primitive Types
There are 5 + 2 primitive types in JS:
```js
let str    = "Hello";         // String
let num    = 42;              // Number (integers and floats both)
let bool   = true;            // Boolean
let x      = undefined;       // Undefined (declared but not assigned)
let y      = null;            // Null (intentional absence of value)
let sym    = Symbol();        // Symbol (ES6, unique identifier)
let big    = 9007199254740991n; // BigInt (ES2020)
```

### Dynamic Typing
JS is dynamically typed — the type of a variable can change at runtime.
```js
let data = 42;
console.log(typeof data); // "number"

data = "Hello";
console.log(typeof data); // "string"
```

### Objects
An object groups related properties and methods together.
```js
let person = {
  name: "John",
  age: 30,
  isAdmin: false,
  greet: function() {
    console.log("Hi, I'm " + this.name);
  }
};

// Accessing properties
console.log(person.name);       // dot notation
console.log(person["age"]);     // bracket notation

// Modifying
person.age = 31;

// Calling a method
person.greet();
```

### Arrays
An ordered list of values. Index starts at 0.
```js
let colors = ["red", "green", "blue"];

console.log(colors[0]);       // "red"
console.log(colors.length);   // 3

colors[3] = "yellow";         // add element
```

### Functions
A reusable block of code.
```js
function greet(name) {
  return "Hello, " + name;
}

let result = greet("John");
console.log(result); // "Hello, John"
```

### Types of Functions
```js
// 1. Function Declaration
function square(n) {
  return n * n;
}

// 2. Function Expression
const cube = function(n) {
  return n * n * n;
};

// 3. Arrow Function (ES6)
const double = (n) => n * 2;

// 4. Anonymous Function (used inline)
setTimeout(function() {
  console.log("done");
}, 1000);
```

---

## SECTION 3: OPERATORS

### Arithmetic Operators
```js
let a = 10, b = 3;

console.log(a + b);   // 13  - Addition
console.log(a - b);   // 7   - Subtraction
console.log(a * b);   // 30  - Multiplication
console.log(a / b);   // 3.33 - Division
console.log(a % b);   // 1   - Modulus (remainder)
console.log(a ** b);  // 1000 - Exponentiation

// Increment / Decrement
let x = 5;
x++;   // post-increment → x = 6
++x;   // pre-increment
x--;   // post-decrement
--x;   // pre-decrement
```

### Assignment Operators
```js
let x = 10;
x += 5;   // x = x + 5  → 15
x -= 3;   // x = x - 3  → 12
x *= 2;   // x = x * 2  → 24
x /= 4;   // x = x / 4  → 6
x %= 4;   // x = x % 4  → 2
x **= 3;  // x = x ** 3 → 8
```

### Comparison Operators
Always return `true` or `false`.
```js
console.log(5 > 3);    // true
console.log(5 < 3);    // false
console.log(5 >= 5);   // true
console.log(5 <= 4);   // false
```

### Equality Operators
```js
// Loose equality (==) — converts types before comparing
console.log(1 == "1");   // true  (type coercion)
console.log(1 == true);  // true

// Strict equality (===) — NO type conversion
console.log(1 === "1");  // false
console.log(1 === 1);    // true

// Inequality
console.log(1 != "1");   // false (loose)
console.log(1 !== "1");  // true  (strict) ← ALWAYS prefer this
```
> **Best Practice:** Always use `===` and `!==` to avoid unexpected bugs.

### Ternary Operator
A shorthand `if/else` in one line.
```js
// Syntax: condition ? valueIfTrue : valueIfFalse

let age = 20;
let type = age >= 18 ? "Adult" : "Minor";
console.log(type); // "Adult"
```

### Logical Operators
```js
// AND (&&) — true only if BOTH are true
console.log(true && true);   // true
console.log(true && false);  // false

// OR (||) — true if AT LEAST ONE is true
console.log(false || true);  // true
console.log(false || false); // false

// NOT (!) — inverts the value
console.log(!true);  // false
console.log(!false); // true
```

### Logical Operators with Non-Booleans
JS uses **short-circuit evaluation** and returns the actual value, not just `true/false`.
```js
// || returns first TRUTHY value, or last value
console.log(undefined || "default");  // "default"
console.log("John" || "default");     // "John"
console.log(0 || false || "value");   // "value"

// && returns first FALSY value, or last value
console.log("John" && "Bob");         // "Bob"
console.log(null && "Bob");           // null

// Practical use: default values
let userColor = null;
let finalColor = userColor || "blue"; // "blue"
```

**Falsy values in JS:** `false`, `0`, `""`, `null`, `undefined`, `NaN`
**Everything else is truthy.**

### Bitwise Operators
Work on individual bits (32-bit integers).
```js
console.log(1 | 2);   // OR  → 3   (01 | 10 = 11)
console.log(1 & 2);   // AND → 0   (01 & 10 = 00)
console.log(1 ^ 2);   // XOR → 3   (01 ^ 10 = 11)
console.log(~1);       // NOT → -2
console.log(1 << 2);  // Left shift  → 4
console.log(8 >> 2);  // Right shift → 2
```
Practical use: permission flags.
```js
const READ    = 4;  // 100
const WRITE   = 2;  // 010
const EXECUTE = 1;  // 001

let myPermission = READ | WRITE; // 110 = 6

// Check if user has READ permission:
if (myPermission & READ) console.log("Can read"); // true
```

### Operator Precedence
Order: `()` → `**` → `* / %` → `+ -` → comparison → logical
```js
let result = 2 + 3 * 4;     // 14 (not 20), * before +
let result2 = (2 + 3) * 4;  // 20, parentheses override
```

### Exercise — Swapping Variables
```js
let a = "red";
let b = "blue";

// Using a temp variable
let temp = a;
a = b;
b = temp;

// ES6 way (destructuring)
[a, b] = [b, a];

console.log(a, b); // "blue" "red"
```

---

## SECTION 4: CONTROL FLOW

### If...else
```js
let hour = 14;

if (hour >= 6 && hour < 12) {
  console.log("Good morning");
} else if (hour >= 12 && hour < 18) {
  console.log("Good afternoon");
} else {
  console.log("Good evening");
}
```

### Switch...case
Used when comparing one variable against many specific values.
```js
let role = "admin";

switch (role) {
  case "admin":
    console.log("Full access");
    break;
  case "moderator":
    console.log("Limited access");
    break;
  default:
    console.log("No access");
}
```
> **Important:** Always add `break` to avoid fall-through.

### For Loop
Used when the number of iterations is known.
```js
for (let i = 0; i < 5; i++) {
  console.log(i); // 0, 1, 2, 3, 4
}

// Looping over an array
let names = ["Alice", "Bob", "Charlie"];
for (let i = 0; i < names.length; i++) {
  console.log(names[i]);
}
```

### While Loop
Used when you loop until a condition becomes false.
```js
let i = 0;
while (i < 5) {
  console.log(i);
  i++;
}
```

### Do...While Loop
Executes the block **at least once** before checking the condition.
```js
let i = 0;
do {
  console.log(i);
  i++;
} while (i < 5);
```

### Infinite Loops
A loop that never ends — always a bug (unless intentional like a server loop).
```js
// DANGER - Don't run this
while (true) {
  // no break → infinite
}

// Safe version with break
let count = 0;
while (true) {
  count++;
  if (count === 5) break;
}
```

### For...in
Iterates over the **keys (properties)** of an object.
```js
let person = { name: "John", age: 30, city: "NY" };

for (let key in person) {
  console.log(key, person[key]);
  // name John
  // age 30
  // city NY
}
```
> **Note:** Also works on arrays but use `for...of` for arrays instead.

### For...of
Iterates over **values** of an iterable (array, string, etc.).
```js
let colors = ["red", "green", "blue"];

for (let color of colors) {
  console.log(color); // red, green, blue
}

// Also works on strings
for (let char of "Hello") {
  console.log(char); // H, e, l, l, o
}
```

### Break and Continue
```js
// break — exits the loop entirely
for (let i = 0; i < 10; i++) {
  if (i === 5) break;
  console.log(i); // 0, 1, 2, 3, 4
}

// continue — skips current iteration, goes to next
for (let i = 0; i < 5; i++) {
  if (i === 2) continue;
  console.log(i); // 0, 1, 3, 4
}
```

### Exercises
```js
// Exercise 1: Max of Two Numbers
function max(a, b) {
  return a > b ? a : b;
}

// Exercise 2: Landscape or Portrait
function isLandscape(width, height) {
  return width > height;
}

// Exercise 3: FizzBuzz
for (let i = 1; i <= 20; i++) {
  if (i % 15 === 0) console.log("FizzBuzz");
  else if (i % 3 === 0) console.log("Fizz");
  else if (i % 5 === 0) console.log("Buzz");
  else console.log(i);
}

// Exercise 4: Demerit Points
function checkSpeed(speed) {
  const limit = 70;
  if (speed < limit + 5) return "Ok";
  let points = Math.floor((speed - limit) / 5);
  if (points >= 12) return "License suspended";
  return points;
}

// Exercise 5: Even and Odd Numbers
function showNumbers(limit) {
  for (let i = 0; i <= limit; i++) {
    if (i % 2 === 0) console.log(i, "EVEN");
    else console.log(i, "ODD");
  }
}

// Exercise 6: Count Truthy
function countTruthy(arr) {
  let count = 0;
  for (let val of arr) {
    if (val) count++;
  }
  return count;
}

// Exercise 7: String Properties
function showProperties(obj) {
  for (let key in obj) {
    if (typeof obj[key] === "string") console.log(key);
  }
}

// Exercise 8: Sum of Multiples of 3 and 5
function sum(limit) {
  let total = 0;
  for (let i = 0; i <= limit; i++) {
    if (i % 3 === 0 || i % 5 === 0) total += i;
  }
  return total;
}

// Exercise 9: Grade
function calculateGrade(marks) {
  let avg = marks.reduce((a, b) => a + b) / marks.length;
  if (avg >= 90) return "A";
  if (avg >= 80) return "B";
  if (avg >= 70) return "C";
  if (avg >= 60) return "D";
  return "F";
}

// Exercise 10: Stars
function showStars(rows) {
  for (let i = 1; i <= rows; i++) {
    console.log("*".repeat(i));
  }
}

// Exercise: Prime Numbers
function showPrimes(limit) {
  for (let i = 2; i <= limit; i++) {
    if (isPrime(i)) console.log(i);
  }
}
function isPrime(n) {
  for (let i = 2; i < n; i++) {
    if (n % i === 0) return false;
  }
  return true;
}
```

---

## SECTION 5: OBJECTS

### Basics
```js
let circle = {
  radius: 1,
  location: { x: 1, y: 1 },          // nested object
  draw: function() {
    console.log("draw");
  }
};

circle.draw();
```

### Factory Functions
A function that creates and returns an object. Used to create multiple similar objects.
```js
function createCircle(radius) {
  return {
    radius,                             // shorthand for radius: radius
    draw() {
      console.log("draw");
    }
  };
}

const c1 = createCircle(1);
const c2 = createCircle(2);
```

### Constructor Functions
Another way to create objects. Use `new` keyword. Convention: PascalCase name.
```js
function Circle(radius) {
  this.radius = radius;
  this.draw = function() {
    console.log("draw");
  };
}

const c = new Circle(5);
// new: creates empty object {}, sets this = {}, returns it
```

### Dynamic Nature of Objects
You can add or delete properties after creation.
```js
const circle = { radius: 1 };

// Add properties
circle.color = "red";
circle.draw = function() {};

// Delete properties
delete circle.color;

console.log(circle); // { radius: 1, draw: [Function] }
```

### Constructor Property
Every object has a `constructor` property pointing to the function that created it.
```js
let x = {};
console.log(x.constructor);         // Object()

let arr = [];
console.log(arr.constructor);       // Array()

function Circle(r) { this.r = r; }
let c = new Circle(1);
console.log(c.constructor);         // Circle()
```

### Functions are Objects
In JS, functions are objects — they have properties and methods.
```js
function Circle(radius) {
  this.radius = radius;
}

console.log(Circle.name);   // "Circle"
console.log(Circle.length); // number of parameters → 1

// .call() and .apply() are methods on function objects
Circle.call({}, 1);
Circle.apply({}, [1]);
```

### Value vs Reference Types
```js
// Primitives are COPIED by value
let a = 10;
let b = a;
b = 20;
console.log(a); // 10 — unchanged

// Objects are COPIED by reference
let x = { val: 10 };
let y = x;
y.val = 20;
console.log(x.val); // 20 — x was also changed!

// Both x and y point to the same object in memory
```

### Enumerating Properties of an Object
```js
const circle = { radius: 1, draw() {} };

// for...in — includes methods too
for (let key in circle) {
  console.log(key, circle[key]);
}

// Object.keys() — returns array of keys
Object.keys(circle).forEach(key => console.log(key));

// Object.entries() — returns [key, value] pairs
Object.entries(circle).forEach(([k, v]) => console.log(k, v));

// Check if property exists
if ("radius" in circle) console.log("exists");
```

### Cloning an Object
```js
const circle = { radius: 1, draw() {} };

// Old way
const another = Object.assign({}, circle);

// Modern way — Spread operator
const clone = { ...circle };

// Add/override during spread
const bigger = { ...circle, radius: 10 };
```

### Garbage Collection
JS has automatic garbage collection. The engine automatically frees memory when objects are no longer referenced. You don't need to manually allocate/deallocate memory like in C/C++.

### Math Object
```js
console.log(Math.PI);           // 3.14159...
console.log(Math.round(4.7));   // 5
console.log(Math.floor(4.9));   // 4
console.log(Math.ceil(4.1));    // 5
console.log(Math.abs(-5));      // 5
console.log(Math.pow(2, 3));    // 8
console.log(Math.sqrt(16));     // 4
console.log(Math.min(1, 2, 3)); // 1
console.log(Math.max(1, 2, 3)); // 3
console.log(Math.random());     // 0.0 to <1.0

// Random integer between min and max
function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}
```

### String
Strings are primitive but JS wraps them in a String object when you access properties.
```js
const msg = "  Hello World  ";

console.log(msg.length);              // 15
console.log(msg.toUpperCase());       // "  HELLO WORLD  "
console.log(msg.toLowerCase());       // "  hello world  "
console.log(msg.trim());              // "Hello World"
console.log(msg.trimStart());         // "Hello World  "
console.log(msg.trimEnd());           // "  Hello World"
console.log(msg.includes("World"));   // true
console.log(msg.startsWith("  He"));  // true
console.log(msg.endsWith("  "));      // true
console.log(msg.indexOf("World"));    // 8
console.log(msg.replace("World","JS")); // "  Hello JS  "
console.log(msg.slice(2, 7));         // "Hello"
console.log(msg.split(" "));          // array of words
```

### Template Literals
Backtick strings that allow embedded expressions and multi-line.
```js
const name = "John";
const age = 30;

// Old way
const msg1 = "My name is " + name + " and I am " + age;

// Template literal (ES6)
const msg2 = `My name is ${name} and I am ${age}`;

// Multi-line
const html = `
  <div>
    <h1>${name}</h1>
  </div>
`;

// Expression inside ${}
console.log(`${2 + 3} is the result`); // "5 is the result"
```

### Date
```js
const now = new Date();                  // current date/time
const d1  = new Date("2020-01-01");      // from string
const d2  = new Date(2020, 0, 1, 10, 0); // year, month(0-based), day

console.log(now.getFullYear());   // 2026
console.log(now.getMonth());      // 0-11
console.log(now.getDate());       // day of month
console.log(now.getDay());        // day of week (0=Sunday)
console.log(now.getHours());      
console.log(now.toDateString());  // "Fri Mar 13 2026"
console.log(now.toISOString());   // "2026-03-13T..."
console.log(now.toLocaleDateString()); // locale format
```

---

## SECTION 6: ARRAYS

### Adding Elements
```js
const arr = [3, 4];

// End
arr.push(5, 6);        // [3, 4, 5, 6]

// Beginning
arr.unshift(1, 2);     // [1, 2, 3, 4, 5, 6]

// Middle — splice(startIndex, deleteCount, ...items)
arr.splice(2, 0, "a", "b"); // insert at index 2 without deleting
```

### Finding Elements (Primitives)
```js
const arr = [1, 2, 3, 4, 3];

console.log(arr.indexOf(3));         // 2 (first occurrence)
console.log(arr.lastIndexOf(3));     // 4
console.log(arr.indexOf(10));        // -1 (not found)
console.log(arr.includes(3));        // true
```

### Finding Elements (Objects)
```js
const courses = [
  { id: 1, name: "JS" },
  { id: 2, name: "Python" }
];

// find() — returns the element itself or undefined
const c = courses.find(c => c.name === "JS");
console.log(c); // { id: 1, name: "JS" }

// findIndex() — returns index or -1
const idx = courses.findIndex(c => c.name === "JS");
console.log(idx); // 0
```

### Arrow Functions
Concise syntax for function expressions.
```js
// Regular function expression
const square = function(n) { return n * n; };

// Arrow function
const square = (n) => { return n * n; };

// If single expression, can omit braces and return
const square = n => n * n;

// No parameters
const greet = () => "Hello";
```

### Removing Elements
```js
const arr = [1, 2, 3, 4];

arr.pop();              // removes last → [1, 2, 3]
arr.shift();            // removes first → [2, 3]
arr.splice(1, 1);       // removes 1 element at index 1 → [2]

// splice returns the removed elements
const removed = arr.splice(0, 2); // removes 2 from index 0
```

### Emptying an Array
```js
let arr = [1, 2, 3, 4];

// Solution 1 — Reassign (only if no other references)
arr = [];

// Solution 2 — Set length to 0 (works even with references)
arr.length = 0;

// Solution 3 — splice
arr.splice(0, arr.length);
```

### Combining and Slicing Arrays
```js
const first = [1, 2, 3];
const second = [4, 5, 6];

// concat — creates a new array
const combined = first.concat(second); // [1,2,3,4,5,6]

// slice — returns a portion (doesn't modify original)
const sliced = combined.slice(2, 5);   // [3, 4, 5]
const copy = combined.slice();          // full copy
```

### Spread Operator
Expands an iterable (array/object) into individual elements.
```js
const first = [1, 2, 3];
const second = [4, 5, 6];

// Combine arrays
const combined = [...first, ...second];          // [1,2,3,4,5,6]
const withMiddle = [...first, "x", ...second];   // with extra items

// Copy
const copy = [...first];

// Pass array as function arguments
Math.max(...first); // same as Math.max(1, 2, 3)
```

### Iterating an Array
```js
const colors = ["red", "green", "blue"];

// for...of (preferred)
for (let color of colors) {
  console.log(color);
}

// forEach (functional style)
colors.forEach((color, index) => {
  console.log(index, color);
});
```

### Joining Arrays
```js
const arr = ["Hello", "World"];

console.log(arr.join(" "));   // "Hello World"
console.log(arr.join("-"));   // "Hello-World"
console.log(arr.join());      // "Hello,World" (default comma)

// Reverse of join — split a string into array
const str = "Hello World";
console.log(str.split(" ")); // ["Hello", "World"]
```

### Sorting Arrays
```js
// Strings — sort alphabetically
const colors = ["blue", "red", "green"];
colors.sort();
console.log(colors); // ["blue", "green", "red"]

colors.reverse(); // reverses in place

// Numbers — must use compare function
const nums = [2, 10, 1, 5];
nums.sort((a, b) => a - b);  // ascending: [1, 2, 5, 10]
nums.sort((a, b) => b - a);  // descending: [10, 5, 2, 1]

// Objects
const courses = [
  { name: "Node" },
  { name: "JavaScript" }
];
courses.sort((a, b) => {
  if (a.name < b.name) return -1;
  if (a.name > b.name) return 1;
  return 0;
});
```

### Testing Elements of an Array
```js
const nums = [1, 2, 3, -1, 4];

// every() — returns true if ALL elements pass the test
const allPositive = nums.every(n => n > 0);
console.log(allPositive); // false (because of -1)

// some() — returns true if AT LEAST ONE element passes
const hasNegative = nums.some(n => n < 0);
console.log(hasNegative); // true
```

### Filtering an Array
`filter()` returns a **new array** with elements that pass the test.
```js
const nums = [1, -2, 3, -4, 5];

const positives = nums.filter(n => n > 0);
console.log(positives); // [1, 3, 5]

const courses = [
  { name: "JS", price: 10 },
  { name: "Python", price: 5 }
];
const cheap = courses.filter(c => c.price <= 5);
```

### Mapping an Array
`map()` transforms every element and returns a **new array** of the same length.
```js
const nums = [1, 2, 3];

const doubled = nums.map(n => n * 2);
console.log(doubled); // [2, 4, 6]

// Map objects to strings
const items = [{ id: 1, name: "A" }, { id: 2, name: "B" }];
const names = items.map(item => item.name);
console.log(names); // ["A", "B"]

// Chain filter + map
const result = nums
  .filter(n => n > 1)
  .map(n => ({ value: n }));
```

### Reducing an Array
`reduce()` reduces an array to a **single value** (sum, product, object, etc.).
```js
const nums = [1, 2, 3, 4, 5];

// accumulator starts at 0, goes through each element
const sum = nums.reduce((acc, curr) => acc + curr, 0);
console.log(sum); // 15

// Without initial value (first element becomes accumulator)
const product = nums.reduce((acc, curr) => acc * curr);
console.log(product); // 120

// Reduce to an object — count occurrences
const votes = ["yes", "no", "yes", "yes", "no"];
const tally = votes.reduce((acc, vote) => {
  acc[vote] = (acc[vote] || 0) + 1;
  return acc;
}, {});
console.log(tally); // { yes: 3, no: 2 }
```

### Array Exercises
```js
// Exercise 1: Array from Range
function arrayFromRange(min, max) {
  const result = [];
  for (let i = min; i <= max; i++) result.push(i);
  return result;
}

// Exercise 2: Includes (without .includes)
function includes(arr, searchElement) {
  for (let element of arr)
    if (element === searchElement) return true;
  return false;
}

// Exercise 3: Except (remove element)
function except(arr, excluded) {
  return arr.filter(e => !excluded.includes(e));
}

// Exercise 4: Moving an Element
function move(arr, index, offset) {
  const output = [...arr];
  const position = index + offset;
  if (position >= output.length || position < 0) {
    console.error("Invalid offset");
    return;
  }
  const element = output.splice(index, 1)[0];
  output.splice(position, 0, element);
  return output;
}

// Exercise 5: Count Occurrences
function countOccurrences(arr, searchElement) {
  return arr.reduce((acc, curr) => {
    return curr === searchElement ? acc + 1 : acc;
  }, 0);
}

// Exercise 6: Get Max
function getMax(arr) {
  if (arr.length === 0) return undefined;
  return arr.reduce((max, curr) => curr > max ? curr : max);
}

// Exercise 7: Movies
const movies = [
  { title: "a", year: 2018, rating: 4.5 },
  { title: "b", year: 2018, rating: 4.7 },
  { title: "c", year: 2018, rating: 3 },
  { title: "d", year: 2017, rating: 4.5 }
];
// Get titles of 2018 movies with rating >= 4, sorted desc by rating
const result = movies
  .filter(m => m.year === 2018 && m.rating >= 4)
  .sort((a, b) => b.rating - a.rating)
  .map(m => m.title);
```

---

## SECTION 7: FUNCTIONS

### Function Declarations vs Expressions
```js
// Declaration — hoisted (can call before definition)
walk();
function walk() {
  console.log("walking");
}

// Expression — NOT hoisted
// run(); // ERROR
const run = function() {
  console.log("running");
};
```

### Hoisting
JS moves **function declarations** to the top of their scope before execution. `var` declarations are also hoisted (but not their assignment). `let` and `const` are NOT hoisted.
```js
// This works due to hoisting
greet();
function greet() { console.log("Hello"); }

// var is hoisted but value is undefined
console.log(x); // undefined (not an error)
var x = 5;

// let — NOT hoisted
// console.log(y); // ReferenceError
let y = 5;
```

### Arguments Object
Every function (non-arrow) has an `arguments` object — an array-like object of all passed arguments.
```js
function sum() {
  let total = 0;
  for (let val of arguments) total += val;
  return total;
}

console.log(sum(1, 2, 3, 4)); // 10
```

### Rest Operator
Collects all remaining arguments into a **real array**. Must be the last parameter.
```js
function sum(...numbers) {
  return numbers.reduce((acc, n) => acc + n, 0);
}

console.log(sum(1, 2, 3, 4, 5)); // 15

// With other params
function log(discount, ...prices) {
  prices.forEach(p => console.log(p * (1 - discount)));
}

log(0.1, 10, 20, 30); // discounted prices
```

### Default Parameters
Provide fallback values when arguments are not passed. Must be at the **end**.
```js
function greet(name = "Guest", greeting = "Hello") {
  return `${greeting}, ${name}!`;
}

console.log(greet());           // "Hello, Guest!"
console.log(greet("John"));     // "Hello, John!"
console.log(greet("John","Hi")); // "Hi, John!"
```

### Getters and Setters
Define computed properties and validate before setting values.
```js
const person = {
  firstName: "John",
  lastName:  "Smith",

  get fullName() {
    return `${this.firstName} ${this.lastName}`;
  },

  set fullName(value) {
    const parts = value.split(" ");
    this.firstName = parts[0];
    this.lastName  = parts[1];
  }
};

console.log(person.fullName);     // "John Smith"
person.fullName = "Alice Brown";  // calls setter
console.log(person.firstName);    // "Alice"
```

### Try and Catch
Handle runtime errors gracefully.
```js
function getColor(id) {
  if (id < 0) throw new Error("ID cannot be negative");
  return "red";
}

try {
  const color = getColor(-1);
  console.log(color);
} catch (err) {
  console.error("Error:", err.message); // "Error: ID cannot be negative"
} finally {
  console.log("Always runs"); // cleanup code
}
```

### Local vs Global Scope
```js
// Global scope — accessible everywhere
let message = "Hello";

function greet() {
  // Local scope — only inside this function
  let localVar = "I'm local";
  console.log(message);   // can access global
  console.log(localVar);  // works
}

greet();
// console.log(localVar); // ERROR — not accessible outside
```

### Let vs Var
```js
// var — function-scoped, hoisted
function example() {
  for (var i = 0; i < 3; i++) {}
  console.log(i); // 3 — var leaks out of the for block
}

// let — block-scoped (preferred)
function example2() {
  for (let i = 0; i < 3; i++) {}
  // console.log(i); // ERROR — let is block-scoped
}

// var attaches to window object globally
var color = "red";
console.log(window.color); // "red" (in browser)

let size = 5;
console.log(window.size); // undefined
```
> **Best Practice:** Always use `let` and `const`. Never use `var`.

### The `this` Keyword
`this` refers to the object that is executing the current function.
```js
// In a method — this = object that owns the method
const video = {
  title: "a",
  play() {
    console.log(this); // { title: "a", play: f }
  }
};
video.play();

// In a regular function — this = global (window in browser) or undefined (strict mode)
function playVideo() {
  console.log(this); // Window (or undefined in strict mode)
}
playVideo();

// In a constructor function — this = new empty object
function Video(title) {
  this.title = title;
  console.log(this); // Video { title: "a" }
}
const v = new Video("a");

// Callback problem — this is lost
const video2 = {
  title: "a",
  tags: ["x", "y"],
  showTags() {
    // Inside forEach callback, this = Window
    this.tags.forEach(tag => {
      console.log(this.title, tag); // Arrow fn fixes this!
    });
  }
};
```

### Changing `this` (call, apply, bind)
```js
function playVideo(a, b) {
  console.log(this.title, a, b);
}

const video = { title: "My Video" };

// call — calls immediately, args passed individually
playVideo.call(video, 1, 2);

// apply — calls immediately, args passed as array
playVideo.apply(video, [1, 2]);

// bind — returns NEW function with this permanently set
const fn = playVideo.bind(video);
fn(1, 2);

// Practical use: fixing 'this' in callbacks
const video2 = {
  title: "a",
  tags: ["x", "y"],
  showTags() {
    this.tags.forEach(function(tag) {
      console.log(this.title, tag);
    }.bind(this)); // bind fixes the this context
  }
};
```

### Function Exercises
```js
// Exercise 1: Sum of Arguments
function sum(...args) {
  if (args.length === 1 && Array.isArray(args[0]))
    return args[0].reduce((a, b) => a + b, 0);
  return args.reduce((a, b) => a + b, 0);
}
console.log(sum(1, 2, 3));        // 6
console.log(sum([1, 2, 3, 4]));   // 10

// Exercise 2: Area of Circle
const circle = {
  radius: 5,
  get area() {
    return Math.PI * this.radius ** 2;
  }
};
console.log(circle.area); // 78.53...

// Exercise 3: Error Handling
function CountError(message) {
  this.message = message;
}

function countOccurrences(arr, searchElement) {
  if (!Array.isArray(arr)) throw new CountError("Not an array");
  return arr.filter(e => e === searchElement).length;
}

try {
  const count = countOccurrences(null, 1);
} catch (e) {
  if (e instanceof CountError)
    console.log("CountError:", e.message);
}
```

---

## QUICK REFERENCE SUMMARY

| Topic | Key Point |
|---|---|
| `let` vs `const` | `let` = reassignable, `const` = fixed reference |
| `==` vs `===` | Always use `===` (strict equality) |
| `var` vs `let` | Never use `var` — use `let`/`const` |
| Arrow functions | Inherit `this` from surrounding scope |
| `for...in` | For object keys |
| `for...of` | For array values |
| `map` | Transform → new array same length |
| `filter` | Filter → new array shorter or equal |
| `reduce` | Collapse → single value |
| `call/apply/bind` | Manually set `this` |
| Spread `...` | Expand array/object |
| Rest `...` | Collect arguments into array |
| Template literals | Use backticks + `${}` |
