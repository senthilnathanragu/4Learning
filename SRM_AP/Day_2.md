# C++ Notes — Part 2

---

## 1. Arithmetic, Relational, and Logical Operators

Already covered in detail in Part 1, so here's a focused recap with new examples.

### Arithmetic
```cpp
int a = 17, b = 5;

cout << a + b;   // 22
cout << a - b;   // 12
cout << a * b;   // 85
cout << a / b;   // 3  ← integer division, decimal part dropped!
cout << a % b;   // 2  ← remainder only

// Fix division to get decimal:
cout << (double)a / b;   // 3.4  ← cast a to double first
```

### Relational — always return true(1) or false(0)
```cpp
int x = 10, y = 20;

cout << (x == y);   // 0  (false)
cout << (x != y);   // 1  (true)
cout << (x < y);    // 1
cout << (x > y);    // 0
cout << (x <= 10);  // 1
cout << (x >= 15);  // 0
```

### Logical — combine multiple conditions
```cpp
int age = 20;
bool hasID = true;

// AND — both must be true
if (age >= 18 && hasID)
    cout << "Entry allowed";

// OR — at least one must be true
if (age < 5 || age > 60)
    cout << "Free ticket";

// NOT — flips the result
if (!hasID)
    cout << "No ID, no entry";
```

### Short-circuit Evaluation
```cpp
// In &&, if first condition is FALSE → second is NOT checked
// In ||, if first condition is TRUE  → second is NOT checked

int x = 0;
if (x != 0 && 10/x > 1)   // safe! division never reached
    cout << "ok";
```

---

## 2. Bitwise Operators

Bitwise operators work directly on **binary bits** of integers. Very important in systems programming, embedded systems, and competitive programming.

### Binary Reminder
```
5  in binary = 0000 0101
12 in binary = 0000 1100
```

### Bitwise Operators Table

| Operator | Name | Description |
|---|---|---|
| `&` | AND | 1 only if BOTH bits are 1 |
| `\|` | OR | 1 if AT LEAST one bit is 1 |
| `^` | XOR | 1 if bits are DIFFERENT |
| `~` | NOT | flips all bits |
| `<<` | Left Shift | shifts bits left (multiply by 2) |
| `>>` | Right Shift | shifts bits right (divide by 2) |

### AND `&`
```
  5 = 0101
  3 = 0011
  --------
  &   0001  = 1
```
```cpp
cout << (5 & 3);    // 1
```

### OR `|`
```
  5 = 0101
  3 = 0011
  --------
  |   0111  = 7
```
```cpp
cout << (5 | 3);    // 7
```

### XOR `^`
```
  5 = 0101
  3 = 0011
  --------
  ^   0110  = 6
```
```cpp
cout << (5 ^ 3);    // 6

// Cool trick: XOR a number with itself = 0
cout << (7 ^ 7);    // 0

// XOR swap (no temp variable needed):
int a = 5, b = 9;
a = a ^ b;
b = a ^ b;
a = a ^ b;
cout << a << " " << b;   // 9 5
```

### NOT `~`
```cpp
cout << (~5);   // -6
// Flips ALL bits including sign bit
// Formula: ~n = -(n+1)
```

### Left Shift `<<`
```cpp
// Shifts bits left by n positions = multiply by 2^n
cout << (1 << 3);   // 1 * 2^3 = 8
cout << (5 << 1);   // 5 * 2   = 10
cout << (3 << 2);   // 3 * 4   = 12

/*
  3 = 0000 0011
  3 << 2:
      0000 1100  = 12
*/
```

### Right Shift `>>`
```cpp
// Shifts bits right by n positions = divide by 2^n
cout << (16 >> 2);   // 16 / 4 = 4
cout << (20 >> 1);   // 20 / 2 = 10

/*
  20 = 0001 0100
  20 >> 1:
       0000 1010  = 10
*/
```

### Practical Uses
```cpp
// Check if a number is even or odd:
if (n & 1)
    cout << "Odd";
else
    cout << "Even";

// Check if specific bit is set (bit position 2):
int flags = 0b1010;
if (flags & (1 << 2))
    cout << "Bit 2 is ON";

// Set a bit ON:
flags = flags | (1 << 1);

// Set a bit OFF:
flags = flags & ~(1 << 1);

// Toggle a bit:
flags = flags ^ (1 << 1);
```

---

## 3. Decision Making: `if`, `else if`, `else`

### Basic `if`
```cpp
int marks = 75;

if (marks >= 50) {
    cout << "You passed!";
}
// If condition is false, nothing happens
```

### `if-else`
```cpp
int age = 16;

if (age >= 18) {
    cout << "You can vote.";
} else {
    cout << "Too young to vote.";
}
```

### `if - else if - else`
```cpp
int marks = 82;

if (marks >= 90) {
    cout << "Grade: A";
} else if (marks >= 80) {
    cout << "Grade: B";
} else if (marks >= 70) {
    cout << "Grade: C";
} else if (marks >= 60) {
    cout << "Grade: D";
} else {
    cout << "Grade: F";
}
```

### Nested `if`
```cpp
int age = 20;
bool citizen = true;

if (age >= 18) {
    if (citizen) {
        cout << "Eligible to vote.";
    } else {
        cout << "Not a citizen.";
    }
} else {
    cout << "Too young.";
}
```

### Ternary Operator — shorthand if-else
```cpp
// Syntax: condition ? value_if_true : value_if_false

int x = 10;
string result = (x % 2 == 0) ? "Even" : "Odd";
cout << result;   // Even

int a = 5, b = 9;
int max = (a > b) ? a : b;
cout << max;   // 9
```

### Common Mistakes
```cpp
// ❌ Assignment instead of comparison:
if (x = 5)    // assigns 5 to x, always true!
if (x == 5)   // ✅ correct comparison

// ❌ Missing braces (only first line is in if):
if (x > 0)
    cout << "positive";
    cout << "done";      // this ALWAYS runs!

// ✅ Always use braces:
if (x > 0) {
    cout << "positive";
    cout << "done";
}
```

---

## 4. Selection Logic: `switch-case`

`switch` is cleaner than long `if-else if` chains when checking **one variable against many fixed values**.

### Syntax
```cpp
switch (expression) {
    case value1:
        // code
        break;
    case value2:
        // code
        break;
    default:
        // runs if no case matches
}
```

### Example — Day of Week
```cpp
int day = 3;

switch (day) {
    case 1:
        cout << "Monday";
        break;
    case 2:
        cout << "Tuesday";
        break;
    case 3:
        cout << "Wednesday";
        break;
    case 4:
        cout << "Thursday";
        break;
    case 5:
        cout << "Friday";
        break;
    default:
        cout << "Weekend";
}
// Output: Wednesday
```

### ⚠️ Fall-through — what happens without `break`
```cpp
int x = 2;
switch (x) {
    case 1:
        cout << "One\n";
    case 2:
        cout << "Two\n";    // ← starts here
    case 3:
        cout << "Three\n";  // ← also runs! (no break above)
    default:
        cout << "Other\n";  // ← also runs!
}
// Output: Two  Three  Other
```

### Intentional Fall-through (grouping cases)
```cpp
char grade = 'B';

switch (grade) {
    case 'A':
    case 'B':
    case 'C':
        cout << "Passing grade";   // runs for A, B, or C
        break;
    case 'D':
    case 'F':
        cout << "Failing grade";
        break;
    default:
        cout << "Invalid grade";
}
```

### switch vs if-else — when to use which

| Use `switch` | Use `if-else` |
|---|---|
| One variable, many fixed values | Range checks (`x > 10`) |
| int, char, enum values | Complex conditions |
| Cleaner, more readable | Boolean/logical checks |

---

## 5. Looping: `for` and `while`

### `for` loop — use when you know how many times to loop
```cpp
// Syntax:
for (initialization; condition; update) {
    // body
}

// Count 1 to 5:
for (int i = 1; i <= 5; i++) {
    cout << i << " ";
}
// Output: 1 2 3 4 5

// Count down:
for (int i = 5; i >= 1; i--) {
    cout << i << " ";
}
// Output: 5 4 3 2 1

// Even numbers:
for (int i = 2; i <= 10; i += 2) {
    cout << i << " ";
}
// Output: 2 4 6 8 10
```

### Nested `for` loops
```cpp
// Multiplication table:
for (int i = 1; i <= 3; i++) {
    for (int j = 1; j <= 3; j++) {
        cout << i * j << "\t";
    }
    cout << endl;
}
/*
Output:
1   2   3
2   4   6
3   6   9
*/

// Star pattern:
for (int i = 1; i <= 4; i++) {
    for (int j = 1; j <= i; j++) {
        cout << "* ";
    }
    cout << endl;
}
/*
*
* *
* * *
* * * *
*/
```

### `while` loop — use when condition is checked first, count unknown
```cpp
// Syntax:
while (condition) {
    // body
    // must update something or infinite loop!
}

// Basic example:
int i = 1;
while (i <= 5) {
    cout << i << " ";
    i++;
}
// Output: 1 2 3 4 5

// User input validation:
int num;
cout << "Enter positive number: ";
cin >> num;
while (num <= 0) {
    cout << "Invalid! Try again: ";
    cin >> num;
}
cout << "You entered: " << num;
```

### `for` vs `while` — same thing, different style
```cpp
// These are identical:

for (int i = 0; i < 5; i++) {
    cout << i;
}

int i = 0;
while (i < 5) {
    cout << i;
    i++;
}
```

### `break` and `continue`
```cpp
// break — exit loop immediately
for (int i = 1; i <= 10; i++) {
    if (i == 5) break;
    cout << i << " ";
}
// Output: 1 2 3 4

// continue — skip current iteration, go to next
for (int i = 1; i <= 10; i++) {
    if (i % 2 == 0) continue;
    cout << i << " ";
}
// Output: 1 3 5 7 9  (odd numbers only)
```

---

## 6. Post-condition Looping: `do-while`

The key difference: **the body runs at LEAST once**, even if the condition is false from the start.

```
       ┌─────────────┐
       │             │
       ▼             │
  [Execute body] ────┘
       │
  [Check condition]
       │
    false → exit
```

### Syntax
```cpp
do {
    // body — runs at least once
} while (condition);   // ← semicolon required!
```

### Example
```cpp
int i = 1;
do {
    cout << i << " ";
    i++;
} while (i <= 5);
// Output: 1 2 3 4 5
```

### Difference between `while` and `do-while`
```cpp
// while — condition checked BEFORE body:
int x = 10;
while (x < 5) {
    cout << "while runs";   // ← NEVER runs
}

// do-while — condition checked AFTER body:
int x = 10;
do {
    cout << "do-while runs";   // ← runs ONCE even though x > 5
} while (x < 5);
```

### Best use case — menu systems
```cpp
int choice;

do {
    cout << "\n--- Menu ---\n";
    cout << "1. Add\n";
    cout << "2. Subtract\n";
    cout << "3. Exit\n";
    cout << "Enter choice: ";
    cin >> choice;

    switch (choice) {
        case 1: cout << "Adding...\n"; break;
        case 2: cout << "Subtracting...\n"; break;
        case 3: cout << "Goodbye!\n"; break;
        default: cout << "Invalid option!\n";
    }

} while (choice != 3);   // keep showing menu until user exits
```

---

## 7. Debugging Basics and Flow Analysis

### What is a Bug?
A **bug** is an error that causes incorrect or unexpected behavior in a program.

### Types of Errors

| Type | When detected | Example |
|---|---|---|
| **Syntax Error** | At compile time | Missing `;`, wrong spelling |
| **Runtime Error** | While running | Divide by zero, infinite loop |
| **Logic Error** | Wrong output | Using `+` instead of `*` |

```cpp
// Syntax Error:
int x = 5     // ❌ missing semicolon → won't compile

// Runtime Error:
int a = 10, b = 0;
cout << a / b;    // ❌ division by zero → crash

// Logic Error:
int area = length + width;   // ❌ should be * not +
// compiles and runs, but gives wrong answer!
```

### Debugging Techniques

**1. Print Statements (simplest)**
```cpp
int x = 10;
cout << "DEBUG: x = " << x << endl;   // check value at this point

for (int i = 0; i < 5; i++) {
    cout << "DEBUG: i = " << i << endl;   // trace loop
    // rest of code
}
```

**2. Trace Tables — manually trace code on paper**

For this code:
```cpp
int x = 1;
while (x <= 3) {
    x = x * 2;
    cout << x;
}
```

| Step | x (before) | x * 2 | x (after) | Output |
|---|---|---|---|---|
| 1 | 1 | 2 | 2 | 2 |
| 2 | 2 | 4 | 4 | 4 |
| 3 | 4 | — | — | exits (4 > 3) |

**3. Rubber Duck Debugging**
Explain your code line-by-line out loud (to anyone, even a rubber duck). You'll often spot the mistake yourself.

**4. Using an IDE Debugger**
- Set **breakpoints** — pause execution at a line
- **Step over** — run one line at a time
- **Watch variables** — see variable values change live

### Common Bugs to Watch For
```cpp
// 1. Off-by-one error:
for (int i = 0; i <= 10; i++)   // runs 11 times, not 10!
for (int i = 0; i < 10; i++)    // ✅ runs exactly 10 times

// 2. Infinite loop:
int i = 0;
while (i < 5) {
    cout << i;
    // forgot i++! runs forever
}

// 3. Wrong operator (= vs ==):
if (x = 0)    // assigns 0, condition is always false
if (x == 0)   // ✅ compares

// 4. Integer division:
double result = 7 / 2;      // result = 3.0 (not 3.5!)
double result = 7.0 / 2;    // ✅ result = 3.5

// 5. Uninitialized variable:
int x;
cout << x;    // garbage value — undefined behavior!
```

### Flow Analysis — tracing execution
Always ask: "what is the value of each variable at each step?"

```cpp
int a = 5, b = 3, c;

c = a + b;        // c = 8
a = c - a;        // a = 3
b = c - b;        // b = 5
                  // a and b are swapped!

cout << a << " " << b;   // 3 5
```

---

## 8. Access Specifiers

Access specifiers control **who can see and use** the members of a class. This is a core concept of **Object-Oriented Programming (OOP)**.

### The Three Access Specifiers

| Specifier | Accessible from inside class | Accessible from outside class | Accessible from subclass (inheritance) |
|---|---|---|---|
| `public` | ✅ Yes | ✅ Yes | ✅ Yes |
| `private` | ✅ Yes | ❌ No | ❌ No |
| `protected` | ✅ Yes | ❌ No | ✅ Yes |

### `public`
```cpp
class Car {
public:
    string brand;    // accessible anywhere
    int speed;

    void drive() {
        cout << brand << " is driving at " << speed << endl;
    }
};

int main() {
    Car c;
    c.brand = "Toyota";   // ✅ works — public
    c.speed = 120;         // ✅ works
    c.drive();             // ✅ works
}
```

### `private`
```cpp
class BankAccount {
private:
    double balance;      // hidden from outside

public:
    void deposit(double amount) {
        if (amount > 0)
            balance += amount;   // ✅ private accessed inside class
    }

    void showBalance() {
        cout << "Balance: " << balance << endl;
    }
};

int main() {
    BankAccount acc;
    acc.balance = 1000;     // ❌ ERROR — private!
    acc.deposit(1000);      // ✅ use public method instead
    acc.showBalance();      // ✅ works
}
```

### `protected`
```cpp
class Animal {
protected:
    string name;   // hidden from outside, but visible to child classes

public:
    Animal(string n) { name = n; }
};

class Dog : public Animal {
public:
    void bark() {
        cout << name << " says Woof!";   // ✅ can access protected member
    }
};

int main() {
    Dog d("Rex");
    d.bark();          // ✅ works
    // d.name = "Max"; // ❌ ERROR — protected, not accessible here
}
```

### Getters and Setters — the right way to access private data
```cpp
class Student {
private:
    string name;
    int age;

public:
    // Setter — set the value
    void setName(string n) {
        name = n;
    }

    void setAge(int a) {
        if (a > 0 && a < 150)    // validation!
            age = a;
        else
            cout << "Invalid age!";
    }

    // Getter — get the value
    string getName() {
        return name;
    }

    int getAge() {
        return age;
    }
};

int main() {
    Student s;
    s.setName("Alice");
    s.setAge(20);

    cout << s.getName() << " is " << s.getAge() << " years old.";
}
```

### Why use `private` + getters/setters?
- **Validation** — check data before storing it
- **Encapsulation** — hide internal details
- **Control** — make some fields read-only (only getter, no setter)
- **Security** — prevent accidental modification

### Default Access in `class` vs `struct`
```cpp
class Foo {
    int x;   // private by default in class
};

struct Bar {
    int x;   // public by default in struct
};
```

---

## Quick Reference Summary

```
Operators:     +  -  *  /  %   |   &  |  ^  ~  <<  >>   |   &&  ||  !
               arithmetic      |   bitwise               |   logical

Decisions:     if / else if / else        switch-case (for fixed values)

Loops:         for   → known count
               while → unknown count, check BEFORE
               do-while → unknown count, run AT LEAST ONCE

Debugging:     Syntax → compile-time   Runtime → crash   Logic → wrong output
               Tools: print statements, trace tables, IDE debugger

Access:        public    → open to all
               private   → class only (use getters/setters)
               protected → class + child classes
```

---

All 8 topics covered in full. Let me know if you want exercises, quizzes, or deeper examples on any section!
