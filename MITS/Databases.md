# 📘 DAY 7 — APPLICATION DESIGN CONCEPTS
## MVC Architecture • Scalability • Memory Limits

---

# ═══════════════════════════════════
# PART 1: MVC (MODEL–VIEW–CONTROLLER) ARCHITECTURE
# ═══════════════════════════════════

## 1.1 What is MVC Architecture?

MVC stands for **Model–View–Controller**. It is a software design pattern used to organize the code in an application by separating it into **three distinct components**, each with a specific responsibility.

Think of MVC like a **restaurant**:

| MVC Layer | Restaurant Analogy | Responsibility in Software |
|---|---|---|
| **Model** | Kitchen (food prep) | Manages data, business logic, database interaction |
| **View** | Dining area (what customer sees) | Displays data to the user — UI, HTML, JSP |
| **Controller** | Waiter (middleman) | Receives user input, talks to Model, updates View |

The core idea of MVC is **Separation of Concerns** — each layer does one job and does it well. A change in the UI (View) does NOT require touching the database code (Model).

> 💡 MVC is used in almost every modern web framework — **Spring MVC** (Java), Django (Python), Laravel (PHP), Ruby on Rails, ASP.NET MVC, and more.

---

## 1.2 MVC Architecture Diagram

```
         User (Browser / Mobile App)
                   │
                   │  HTTP Request (clicks button, submits form)
                   ▼
        ┌─────────────────────────┐
        │       CONTROLLER        │  ← Receives request, decides what to do
        │   (StudentServlet.java) │
        └────────┬────────────────┘
                 │
        ┌────────▼────────┐
        │                 │
        ▼                 ▼
┌──────────────┐   ┌──────────────────┐
│    MODEL     │   │      VIEW        │
│ (Data/Logic) │   │  (UI / Display)  │
│ StudentDAO   │   │ studentList.jsp  │
└──────┬───────┘   └────────▲─────────┘
       │                    │
       │  query / save      │  pass data to display
       ▼                    │
┌──────────────┐            │
│   DATABASE   │────────────┘
│  (MySQL etc) │   model fetches rows → controller passes to view
└──────────────┘
```

**Flow of a request:**
1. User clicks "View All Students" → browser sends request to Controller
2. Controller calls `StudentDAO.getAllStudents()` on the Model
3. Model queries the database and returns a list of Student objects
4. Controller passes the list to the View (JSP)
5. View renders the list as an HTML table and sends it back to the user

---

## 1.3 The Three Layers in Detail

### 1.3.1 MODEL — The Data Layer

The Model represents the **data and business logic** of the application. It is responsible for:
- Connecting to the database (via JDBC)
- Performing CRUD operations (Create, Read, Update, Delete)
- Enforcing business rules (e.g. a student cannot enroll if fees are unpaid)
- Returning data to the Controller — **NOT to the View directly**

**Java Example — Student.java (POJO / Bean):**

```java
// Student.java — Model (Plain Old Java Object)
public class Student {

    private int    id;
    private String name;
    private String email;
    private String city;

    // Constructor
    public Student(int id, String name, String email, String city) {
        this.id    = id;
        this.name  = name;
        this.email = email;
        this.city  = city;
    }

    // Getters
    public int    getId()    { return id;    }
    public String getName()  { return name;  }
    public String getEmail() { return email; }
    public String getCity()  { return city;  }

    // Setters
    public void setName(String name)   { this.name  = name;  }
    public void setEmail(String email) { this.email = email; }
    public void setCity(String city)   { this.city  = city;  }
}
```

**Java Example — StudentDAO.java (handles DB operations):**

```java
// StudentDAO.java — Model (Database Access Object)
import java.sql.*;
import java.util.*;

public class StudentDAO {

    private Connection conn;

    public StudentDAO(Connection conn) {
        this.conn = conn;
    }

    // READ — Fetch all students
    public List<Student> getAllStudents() throws SQLException {
        List<Student> list = new ArrayList<>();
        String sql = "SELECT * FROM students";
        Statement st = conn.createStatement();
        ResultSet rs = st.executeQuery(sql);
        while (rs.next()) {
            list.add(new Student(
                rs.getInt("id"),
                rs.getString("name"),
                rs.getString("email"),
                rs.getString("city")
            ));
        }
        return list;
    }

    // CREATE — Add new student
    public void addStudent(Student s) throws SQLException {
        String sql = "INSERT INTO students (name, email, city) VALUES (?, ?, ?)";
        PreparedStatement ps = conn.prepareStatement(sql);
        ps.setString(1, s.getName());
        ps.setString(2, s.getEmail());
        ps.setString(3, s.getCity());
        ps.executeUpdate();
    }

    // DELETE — Remove a student
    public void deleteStudent(int id) throws SQLException {
        String sql = "DELETE FROM students WHERE id = ?";
        PreparedStatement ps = conn.prepareStatement(sql);
        ps.setInt(1, id);
        ps.executeUpdate();
    }
}
```

---

### 1.3.2 VIEW — The Presentation Layer

The View is what the **user sees** — the User Interface. In Java web applications, Views are usually JSP (Java Server Pages) or HTML files. The View:
- **Displays** data passed from the Controller
- Contains **NO business logic** and **NO database code**
- Receives a ready-made list/object and just renders it as HTML
- When the user interacts (e.g. submits a form), sends a new request back to the Controller

**Java Example — studentList.jsp (View):**

```jsp
<!-- studentList.jsp — View -->
<%@ page import="java.util.*, model.Student" %>
<!DOCTYPE html>
<html>
<head><title>Student List</title></head>
<body>
  <h1>All Students</h1>
  <table border="1">
    <tr>
      <th>ID</th><th>Name</th><th>Email</th><th>City</th>
    </tr>

    <%
      // Data is passed from the Controller
      List<Student> students = (List<Student>) request.getAttribute("studentList");
      for (Student s : students) {
    %>
    <tr>
      <td><%= s.getId()    %></td>
      <td><%= s.getName()  %></td>
      <td><%= s.getEmail() %></td>
      <td><%= s.getCity()  %></td>
    </tr>
    <% } %>
  </table>

  <a href="addStudent.jsp">+ Add New Student</a>
</body>
</html>
```

---

### 1.3.3 CONTROLLER — The Logic Layer

The Controller acts as the **middleman** between the Model and View. It:
- Receives the user's HTTP request (form submission, button click, URL visit)
- Decides what data is needed — calls the appropriate Model method
- Passes the retrieved data to the correct View for display
- Contains **NO HTML/UI code** and **NO direct SQL** — just routing logic

**Java Example — StudentServlet.java (Controller):**

```java
// StudentServlet.java — Controller
import jakarta.servlet.*;
import jakarta.servlet.http.*;
import java.io.*;
import java.sql.*;
import java.util.*;
import model.*;

@WebServlet("/students")
public class StudentServlet extends HttpServlet {

    // Handle GET request → show all students
    @Override
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response)
                         throws ServletException, IOException {
        try {
            // 1. Connect to database
            Connection conn = DriverManager.getConnection(
                "jdbc:mysql://localhost:3306/college_db", "root", "password"
            );

            // 2. Call MODEL to get data
            StudentDAO dao = new StudentDAO(conn);
            List<Student> students = dao.getAllStudents();

            // 3. Pass data to VIEW
            request.setAttribute("studentList", students);
            RequestDispatcher rd = request.getRequestDispatcher("studentList.jsp");
            rd.forward(request, response);

            conn.close();

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    // Handle POST request → add new student
    @Override
    protected void doPost(HttpServletRequest request,
                          HttpServletResponse response)
                          throws ServletException, IOException {
        // 1. Read form data from VIEW
        String name  = request.getParameter("name");
        String email = request.getParameter("email");
        String city  = request.getParameter("city");

        try {
            Connection conn = DriverManager.getConnection(
                "jdbc:mysql://localhost:3306/college_db", "root", "password"
            );

            // 2. Call MODEL to save data
            StudentDAO dao = new StudentDAO(conn);
            dao.addStudent(new Student(0, name, email, city));

            conn.close();

        } catch (SQLException e) {
            e.printStackTrace();
        }

        // 3. Redirect back to student list (VIEW)
        response.sendRedirect("students");
    }
}
```

---

## 1.4 Complete MVC Request-Response Cycle (Step-by-Step)

```
Step 1:  User visits → http://localhost:8080/students
         Browser sends GET request to StudentServlet

Step 2:  StudentServlet (Controller) receives the request
         Calls StudentDAO.getAllStudents() on the Model

Step 3:  StudentDAO (Model) executes:
         SELECT * FROM students
         Returns List<Student> to Controller

Step 4:  Controller does:
         request.setAttribute("studentList", students)
         Forwards to studentList.jsp (View)

Step 5:  studentList.jsp (View) reads the studentList
         Renders an HTML table with all student rows
         Sends final HTML back to the user's browser

Step 6:  User sees the table. Clicks "Add New Student"
         Fills the form and submits → POST request to Controller
         Cycle repeats
```

---

## 1.5 MVC in Real-World Projects

| Layer | File/Class | Technology |
|---|---|---|
| Model | `Student.java`, `StudentDAO.java` | Java, JDBC, SQL |
| View | `studentList.jsp`, `addStudent.jsp` | JSP, HTML, CSS |
| Controller | `StudentServlet.java` | Java Servlet |

**Folder structure of a Java MVC project:**

```
MyApp/
├── src/
│   ├── model/
│   │   ├── Student.java          ← Model (Bean)
│   │   └── StudentDAO.java       ← Model (DB logic)
│   └── controller/
│       └── StudentServlet.java   ← Controller
├── web/
│   ├── studentList.jsp           ← View
│   ├── addStudent.jsp            ← View
│   └── WEB-INF/
│       └── web.xml               ← Servlet mapping config
└── lib/
    └── mysql-connector.jar       ← JDBC Driver
```

---

## 1.6 Advantages & Disadvantages of MVC

### ✅ Advantages:
- **Separation of Concerns** — Each layer is independent; changes in one don't break others
- **Easier Maintenance** — A frontend developer works only on Views; a backend developer works only on Model/Controller
- **Code Reusability** — Same Model can serve multiple Views (web page AND mobile app)
- **Parallel Development** — UI and backend teams can work simultaneously
- **Testability** — Each layer can be unit-tested independently
- **Scalability** — Easy to add new features without restructuring everything

### ❌ Disadvantages:
- **More files and structure** — Overkill for tiny applications
- **Learning curve** — Beginners find it confusing initially
- **Tight coupling between Controller and View** — In classic MVC, the controller is still aware of which view to call
- **Increased complexity** — Simple tasks require creating multiple files

---

## 1.7 Variations of MVC

| Pattern | Full Form | Key Difference | Used In |
|---|---|---|---|
| **MVC** | Model-View-Controller | Controller handles all input | Java Servlets, Spring MVC |
| **MVP** | Model-View-Presenter | Presenter has no direct View reference | Android Apps |
| **MVVM** | Model-View-ViewModel | ViewModel exposes data streams | Angular, Vue.js |
| **MVT** | Model-View-Template | Template = View, no Controller class | Django (Python) |

> 💡 For your exams and Java projects — **MVC is the most important pattern to know**.

---
---

# ═══════════════════════════════════
# PART 2: SCALABILITY
# ═══════════════════════════════════

## 2.1 What is Scalability?

**Scalability** is the ability of an application or system to **handle increasing load** (more users, more data, more requests) **without degrading performance**.

A scalable system can grow to handle 10x or 100x more work by adding resources, without rewriting the entire application.

**Real-world example:**
- IRCTC on a normal day: 1 lakh users → works fine
- IRCTC on Tatkal booking day: 30 lakh users → system crashes without scalability

This is why scalability is not optional — it is a **design requirement** for any production application.

---

## 2.2 Types of Scalability

### 2.2.1 Vertical Scaling (Scale Up)

**Add more power to the existing server** — more CPU, more RAM, faster storage.

```
BEFORE:                          AFTER (Scale Up):
┌──────────────┐                ┌──────────────────────┐
│  Server      │                │  Upgraded Server     │
│  CPU:  4 cores│    ───────►   │  CPU:  32 cores      │
│  RAM:  8 GB  │                │  RAM:  128 GB        │
│  SSD:  500 GB│                │  SSD:  4 TB NVMe     │
└──────────────┘                └──────────────────────┘
  Handles 1,000 req/sec           Handles 10,000 req/sec
```

**Advantages:** Simple, no code changes needed, works immediately.

**Disadvantages:**
- Has a **hardware ceiling** — you can't upgrade a server forever
- **Single point of failure** — if that one server crashes, everything goes down
- **Expensive** — high-end server hardware costs a fortune

---

### 2.2.2 Horizontal Scaling (Scale Out)

**Add more servers** and distribute the load across them. This is how Google, Amazon, Netflix scale.

```
BEFORE:                         AFTER (Scale Out):
                                        ┌──────────────┐
┌──────────────┐                ┌──────►│  Server 1    │
│  Server      │                │       └──────────────┘
│  1 machine   │  ──────────►   │Load   ┌──────────────┐
│              │                │Balancer│  Server 2   │
└──────────────┘                │       └──────────────┘
                                │       ┌──────────────┐
                                └──────►│  Server 3    │
                                        └──────────────┘
```

**Advantages:**
- **No hardware ceiling** — just keep adding servers
- **No single point of failure** — if one server dies, others handle the traffic
- **Cost-effective** — use cheap commodity servers instead of one super-server

**Disadvantages:**
- **Application must be stateless** — sessions can't be stored on one machine
- **More complex** — needs a Load Balancer, distributed session management, etc.
- **Data consistency challenges** — keeping all servers in sync is harder

---

## 2.3 Vertical vs Horizontal Scaling — Comparison

| Feature | Vertical Scaling | Horizontal Scaling |
|---|---|---|
| Method | Upgrade one server | Add more servers |
| Limit | Hardware ceiling | Virtually unlimited |
| Cost | Very expensive | Cheaper (commodity hardware) |
| Downtime | Required for upgrade | None (add servers on the fly) |
| Failure risk | High (single point of failure) | Low (redundant servers) |
| Complexity | Simple | More complex |
| Best for | Small–medium apps, databases | Large-scale web apps, microservices |
| Example | Upgrading MySQL server RAM | Netflix adding 100 more servers |

---

## 2.4 Load Balancing

A **Load Balancer** is a component that sits in front of multiple servers and **distributes incoming requests** evenly among them so no single server is overloaded.

```
                         ┌───────────────────┐
User Requests ──────────►│   LOAD BALANCER   │
                         └──────────┬────────┘
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌──────────┐   ┌──────────┐   ┌──────────┐
             │ Server 1 │   │ Server 2 │   │ Server 3 │
             │ 33% load │   │ 33% load │   │ 34% load │
             └──────────┘   └──────────┘   └──────────┘
```

**Load Balancing Algorithms:**

| Algorithm | How it works | Best for |
|---|---|---|
| **Round Robin** | Sends each request to the next server in rotation | Equal capacity servers |
| **Least Connections** | Sends to the server with fewest active requests | Varying request lengths |
| **IP Hash** | Same user always goes to same server | Session-heavy apps |
| **Weighted Round Robin** | Powerful servers get more requests | Mixed capacity servers |

---

## 2.5 Caching for Scalability

**Caching** means storing frequently accessed data in a **fast memory store** (like Redis) so you don't hit the database every time.

```
WITHOUT CACHE:                     WITH CACHE:
User Request                       User Request
     │                                  │
     ▼                                  ▼
  Application                       Check Cache (Redis)
     │                                  │
     ▼                            ┌─────┴─────┐
  Database  ← slow query!         │           │
     │                        Cache HIT   Cache MISS
     ▼                            │           │
  Response (slow)               Response   Database
                                 (fast!)     │
                                           Cache it
                                             │
                                           Response
```

**Java Example — Checking cache before hitting DB:**

```java
import redis.clients.jedis.Jedis;
import java.sql.*;

public class ProductService {

    private Jedis jedis = new Jedis("localhost", 6379);
    private Connection conn; // SQL connection

    public String getProduct(int productId) throws SQLException {

        String cacheKey = "product:" + productId;

        // Step 1: Check cache first (fast — microseconds)
        String cached = jedis.get(cacheKey);
        if (cached != null) {
            System.out.println("Cache HIT — serving from Redis");
            return cached;
        }

        // Step 2: If not cached, query database (slow — milliseconds)
        System.out.println("Cache MISS — querying database");
        String sql = "SELECT * FROM products WHERE id = ?";
        PreparedStatement ps = conn.prepareStatement(sql);
        ps.setInt(1, productId);
        ResultSet rs = ps.executeQuery();

        String result = "";
        if (rs.next()) {
            result = rs.getString("name") + " - Rs." + rs.getDouble("price");
        }

        // Step 3: Store in cache for 10 minutes (600 seconds)
        jedis.setex(cacheKey, 600, result);

        return result;
    }
}
```

---

## 2.6 Database Scalability Techniques

### 2.6.1 Database Replication
**Copy the database across multiple servers.** One server handles writes (master), others handle reads (slaves).

```
                   ┌──────────────┐
  WRITE requests ─►│  MASTER DB   │──── replicates to ────┐
                   └──────────────┘                        │
                                          ┌────────────────▼──┐
  READ requests ──────────────────────────►   SLAVE DB 1      │
  READ requests ──────────────────────────►   SLAVE DB 2      │
                                          └────────────────────┘
```

**Why?** Most apps have 80% reads and 20% writes. Offloading reads to slaves dramatically reduces load on the master.

### 2.6.2 Database Sharding
**Split the database into smaller pieces called shards** and store each shard on a different server.

```
All Users (10 crore records)
         │
         ▼ Shard by user_id
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  Shard 1       │  │  Shard 2       │  │  Shard 3       │
│  user_id 1–33M │  │  user_id 33–66M│  │  user_id 66–99M│
│  (Server A)    │  │  (Server B)    │  │  (Server C)    │
└────────────────┘  └────────────────┘  └───────────────