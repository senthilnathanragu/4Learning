## 1. Overview of Programming Languages

A **programming language** is a formal set of instructions used to communicate with a computer.

### Types of Programming Languages

**By Level of Abstraction:**
- **Machine Language** — raw binary (0s and 1s), directly understood by CPU. Extremely fast but unreadable.
- **Assembly Language** — uses mnemonics like `MOV`, `ADD`. Still low-level, hardware-specific.
- **High-Level Languages** — C, C++, Java, Python. Human-readable, portable, easier to write.

**By Paradigm:**
- **Procedural** — step-by-step instructions (C, Pascal)
- **Object-Oriented** — organized around objects and classes (C++, Java)
- **Functional** — based on mathematical functions (Haskell, Lisp)
- **Scripting** — interpreted, used for automation (Python, JavaScript)

### Compiled vs Interpreted
| Compiled | Interpreted |
|---|---|
| Translated to machine code before running | Translated line-by-line at runtime |
| Faster execution | Slower but more flexible |
| C, C++, Rust | Python, JavaScript, Ruby |

---

## 2. History & Features of C++

### History
- **1972** — C language created by **Dennis Ritchie** at Bell Labs
- **1979** — **Bjarne Stroustrup** started developing "C with Classes"
- **1983** — Renamed to **C++** (`++` = increment, meaning an improvement over C)
- **1998** — First ISO standard: **C++98**
- Later standards: **C++11**, C++14, C++17, **C++20** (most widely used today)

### Why C++?
C++ was designed to keep C's performance while adding **Object-Oriented Programming** capabilities.

### Key Features of C++

| Feature | Explanation |
|---|---|
| **Compiled Language** | Translates to fast machine code |
| **Statically Typed** | Variable types are declared and checked at compile time |
| **Object-Oriented** | Supports classes, objects, inheritance, polymorphism |
| **Multi-paradigm** | Can write procedural OR object-oriented code |
| **Low-level Access** | Can manipulate memory directly (pointers) |
| **Standard Library (STL)** | Rich built-in library of data structures & algorithms |
| **Portability** | Write once, compile on many platforms |
| **Performance** | Used in games, OS, embedded systems, browsers |

### Where is C++ used?
- Game engines (Unreal Engine)
- Operating systems (Windows internals)
- Browsers (Chrome's V8 engine)
- Databases (MySQL)
- Competitive programming

---

## 3. IDE & Compilation Process

### What is an IDE?
An **Integrated Development Environment** bundles a code editor, compiler, and debugger in one tool.

**Popular IDEs for C++:**
- **VS Code** + MinGW/GCC (lightweight, popular)
- **Code::Blocks** (beginner-friendly)
- **Dev-C++** (simple, Windows)
- **CLion** (professional, paid)
- **Visual Studio** (Windows, full-featured)

### The Compilation Process — Step by Step

When you write C++ code and hit "Run", several stages happen:

```
Source Code (.cpp)
      ↓
[1] Preprocessor     → expands #include, #define macros
      ↓
[2] Compiler         → translates to Assembly code
      ↓
[3] Assembler        → converts to Object code (.obj / .o)
      ↓
[4] Linker           → combines object files + libraries
      ↓
Executable (.exe / a.out)
```

**In simple terms:**
1. **Preprocessor** — handles lines starting with `#` (like `#include <iostream>`)
2. **Compiler** — checks syntax, converts C++ → machine instructions
3. **Linker** — connects your code with library functions like `cout`

### Compiling from Terminal (GCC)
```bash
g++ hello.cpp -o hello     # compile
./hello                    # run (Linux/Mac)
hello.exe                  # run (Windows)
```

---

## 4. Structure of a C++ Program

```cpp
// 1. Preprocessor Directive
#include <iostream>

// 2. Namespace declaration
using namespace std;

// 3. Main function — entry point of every C++ program
int main() {

    // 4. Statements
    cout << "Hello, World!" << endl;

    // 5. Return statement
    return 0;
}
```

### Breaking it down:

| Part | Purpose |
|---|---|
| `#include <iostream>` | Includes the input/output library |
| `using namespace std;` | So we can write `cout` instead of `std::cout` |
| `int main()` | The function where program execution begins |
| `{ }` | Curly braces define a block of code |
| `cout << ...` | Prints to screen |
| `return 0;` | Tells OS the program ended successfully |
| `;` | Every statement ends with a semicolon |
| `//` | Single-line comment |
| `/* */` | Multi-line comment |

### Key Rules:
- Every C++ program **must** have a `main()` function
- Execution always **starts** from `main()`
- Statements end with `;`
- C++ is **case-sensitive** (`Main` ≠ `main`)

---

## 5. Primitive Data Types and Variables

### What is a Variable?
A variable is a **named memory location** that stores a value.

```cpp
int age = 20;       // 'age' is a variable of type int, storing value 20
```

### Declaring Variables
```cpp
datatype variableName;           // declaration
datatype variableName = value;   // declaration + initialization

int x;          // declared (garbage value inside)
int x = 5;      // declared and initialized
```

### Primitive Data Types

| Type | Size | Range / Use | Example |
|---|---|---|---|
| `int` | 4 bytes | Whole numbers: -2B to +2B | `int age = 21;` |
| `short` | 2 bytes | Smaller whole numbers | `short x = 100;` |
| `long` | 4-8 bytes | Larger whole numbers | `long pop = 1000000;` |
| `float` | 4 bytes | Decimal numbers (~6-7 digits precision) | `float pi = 3.14f;` |
| `double` | 8 bytes | Larger decimals (~15 digits precision) | `double g = 9.81;` |
| `char` | 1 byte | Single character | `char grade = 'A';` |
| `bool` | 1 byte | true or false | `bool pass = true;` |
| `void` | — | No value (used in functions) | — |

### `sizeof()` operator
```cpp
cout << sizeof(int);      // prints 4
cout << sizeof(double);   // prints 8
```

### Type Modifiers
```cpp
unsigned int x = 300;    // only positive numbers, bigger range
signed int y = -5;       // default, can be negative
long long z = 9999999999LL;
```

### Variable Naming Rules
- Can contain letters, digits, underscore `_`
- Cannot start with a digit
- Cannot use reserved keywords (`int`, `return`, etc.)
- Case-sensitive (`Score` ≠ `score`)

```cpp
int myAge;       ✅
int 2cool;       ❌ (starts with digit)
int my-age;      ❌ (hyphen not allowed)
int int;         ❌ (reserved keyword)
```

---

## 6. Operators and Expressions

An **expression** is a combination of values, variables, and operators that produces a result.

### Arithmetic Operators
```cpp
int a = 10, b = 3;

cout << a + b;   // 13  (addition)
cout << a - b;   // 7   (subtraction)
cout << a * b;   // 30  (multiplication)
cout << a / b;   // 3   (integer division — truncates!)
cout << a % b;   // 1   (modulus — remainder)
```

> ⚠️ **Important:** `10 / 3 = 3` (not 3.33) when both are `int`. Use `10.0 / 3` for decimal result.

### Assignment Operators
```cpp
int x = 10;
x += 5;    // x = x + 5  → 15
x -= 3;    // x = x - 3  → 12
x *= 2;    // x = x * 2  → 24
x /= 4;    // x = x / 4  → 6
x %= 4;    // x = x % 4  → 2
```

### Increment / Decrement
```cpp
int x = 5;
x++;    // post-increment: use x, then add 1
++x;    // pre-increment: add 1, then use x
x--;    // post-decrement
--x;    // pre-decrement

// Difference matters in expressions:
int a = 5;
cout << a++;   // prints 5, then a becomes 6
cout << ++a;   // a becomes 7, prints 7
```

### Relational (Comparison) Operators
```cpp
// All return true (1) or false (0)
5 == 5    // true  — equal to
5 != 3    // true  — not equal to
5 > 3     // true  — greater than
5 < 3     // false — less than
5 >= 5    // true  — greater than or equal
5 <= 4    // false — less than or equal
```

### Logical Operators
```cpp
// && → AND: both must be true
(5 > 3 && 10 > 7)    // true

// || → OR: at least one must be true
(5 > 3 || 10 < 7)    // true

// ! → NOT: flips true/false
!(5 == 5)             // false
```

### Operator Precedence (high to low)
```
()          → Parentheses first
++ --       → Increment/Decrement
* / %       → Multiplication, Division, Modulus
+ -         → Addition, Subtraction
< > <= >=   → Relational
== !=       → Equality
&&          → Logical AND
||          → Logical OR
=           → Assignment (last)
```

---

## 7. Constants and Literal Values

### What is a Constant?
A constant is a value that **cannot change** after it's set.

### Method 1: `const` keyword (preferred)
```cpp
const double PI = 3.14159;
const int MAX_SIZE = 100;
const char GRADE = 'A';

PI = 3.0;    // ❌ ERROR — cannot modify a constant
```

### Method 2: `#define` (preprocessor macro)
```cpp
#define PI 3.14159
#define MAX 100

// No type checking, just text substitution — less safe
```

### Literal Values
Literals are **fixed values written directly** in code.

```cpp
42          // integer literal
3.14        // double literal
3.14f       // float literal (note the f)
'A'         // character literal (single quotes)
"Hello"     // string literal (double quotes)
true        // boolean literal
false       // boolean literal

// Special integer literals:
0b1010      // binary   = 10
010         // octal    = 8  (starts with 0)
0xFF        // hex      = 255 (starts with 0x)
```

### Escape Characters in literals
```cpp
'\n'    // newline
'\t'    // tab
'\\'    // backslash
'\''    // single quote
'\"'    // double quote
'\0'    // null character
```

---

## 8. Standard Input/Output using `cin` and `cout`

### Output — `cout`
`cout` = **C**haracter **Out**put. Uses `<<` (insertion operator).

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "Hello!";               // basic output
    cout << "Hello!" << endl;       // with newline
    cout << "Hello!" << "\n";       // also newline (faster than endl)
    
    int age = 21;
    cout << "Age: " << age << endl;  // print variable
    
    // Multiple values:
    cout << "Sum of " << 3 << " and " << 4 << " is " << 7 << endl;
}
```

**`endl` vs `"\n"`:**
- Both go to next line
- `endl` also **flushes the buffer** (slightly slower)
- `"\n"` is preferred for performance

### Input — `cin`
`cin` = **C**haracter **In**put. Uses `>>` (extraction operator).

```cpp
int age;
cout << "Enter your age: ";
cin >> age;
cout << "You are " << age << " years old." << endl;
```

```cpp
// Multiple inputs at once:
int a, b;
cin >> a >> b;    // user types: 5 10

// String input (single word):
string name;
cin >> name;      // reads until space

// Full line input:
string fullName;
getline(cin, fullName);    // reads entire line including spaces
```

### Formatting Output
```cpp
#include <iomanip>    // needed for formatting

cout << fixed << setprecision(2) << 3.14159;   // prints: 3.14
cout << setw(10) << "Hello";    // right-align in 10-char wide field
```

### Complete Example:
```cpp
#include <iostream>
using namespace std;

int main() {
    string name;
    int age;
    double gpa;

    cout << "Enter name: ";
    cin >> name;

    cout << "Enter age: ";
    cin >> age;

    cout << "Enter GPA: ";
    cin >> gpa;

    cout << "\n--- Student Info ---" << endl;
    cout << "Name: " << name << endl;
    cout << "Age : " << age << endl;
    cout << "GPA : " << gpa << endl;

    return 0;
}
```

---

## 9. Namespaces and Scope Resolution (`::`)

### What is a Namespace?
A **namespace** is a container that groups related code to avoid **naming conflicts**.

Imagine two libraries both defining a function called `sort()`. Namespaces let you tell the compiler *which* `sort()` you mean.

### The `std` Namespace
All standard C++ library features (like `cout`, `cin`, `string`) live inside the **`std`** namespace.

```cpp
// Without 'using namespace std':
std::cout << "Hello" << std::endl;
std::cin >> x;
std::string name;

// With 'using namespace std':
using namespace std;
cout << "Hello" << endl;
cin >> x;
string name;
```

### Scope Resolution Operator `::`
`::` tells the compiler **which namespace or class** something belongs to.

```cpp
std::cout    // cout from the std namespace
std::cin     // cin from the std namespace
std::endl    // endl from the std namespace
```

### Creating Your Own Namespace
```cpp
#include <iostream>
using namespace std;

namespace Math {
    double PI = 3.14159;
    
    int square(int x) {
        return x * x;
    }
}

namespace Physics {
    double PI = 3.14159265358979;   // more precise version
}

int main() {
    cout << Math::PI << endl;       // 3.14159
    cout << Physics::PI << endl;    // 3.14159265358979
    cout << Math::square(5) << endl; // 25
    return 0;
}
```

### `using` declarations
```cpp
// Use everything from std:
using namespace std;

// Use only specific things (better practice):
using std::cout;
using std::cin;
using std::endl;
```

> ⚠️ **Best Practice:** Avoid `using namespace std;` in large/professional projects — it can cause conflicts. Use `std::cout` explicitly, or import only what you need.

### Nested Namespaces (C++17)
```cpp
namespace Company::Department::Team {
    void work() { }
}
// Access: Company::Department::Team::work();
```

---

## Quick Reference — Cheat Sheet

```cpp
#include <iostream>
using namespace std;

int main() {
    // Variables
    int x = 10;
    double pi = 3.14;
    char c = 'A';
    bool flag = true;
    
    // Constant
    const int MAX = 100;
    
    // Input / Output
    cout << "Enter x: ";
    cin >> x;
    cout << "x = " << x << endl;
    
    // Operators
    int sum = x + 5;
    int rem = x % 3;
    x++;              // increment
    bool result = (x > 5 && x < 20);
    
    // Scope resolution
    std::cout << "Using std explicitly" << std::endl;
    
    return 0;
}
```

---
