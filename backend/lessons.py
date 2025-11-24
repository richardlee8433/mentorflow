# === NEW: Lesson 4 & 5 JSON config ===

LESSON_4 = {
    "id": "lesson4",
    "title": "Project Management Foundations",
    "units": [
        {
            "id": "lesson4_unit4_1",
            "lesson": 4,
            "unit": "4.1",
            "title": "What is a project?",
            "material": (
                "A project is a temporary endeavor with a defined beginning and end. "
                "It creates a unique product, service, or result. "
                "Projects end when objectives are achieved, when needs disappear, "
                "when resources are unavailable, or when laws or strategy change."
            ),
            "question": "Why is a project considered temporary?",
            "key_points": [
                "A project has a defined beginning and end",
                "A project is not ongoing like operations",
            ],
            "min_points": 1,  # ⭐ 初階題：只要抓到一個重點就給分
            "next_unit": "lesson4_unit4_2",
        },
        {
            "id": "lesson4_unit4_2",
            "lesson": 4,
            "unit": "4.2",
            "title": "Projects drive change",
            "material": (
                "Projects drive change. They move an organization from a current state "
                "to a desired future state. They perform one of four change actions: "
                "Move, Add, Change, Delete (MACD). Some projects require a transition state, "
                "such as phased IT upgrades."
            ),
            "question": "What are the four types of change that projects create?",
            "key_points": ["Move", "Add", "Change", "Delete"],
            "min_points": 2,  # 至少說出兩個 MACD 就算過
            "next_unit": "lesson4_unit4_3",
        },
        {
            "id": "lesson4_unit4_3",
            "lesson": 4,
            "unit": "4.3",
            "title": "Why projects are created (business value)",
            "material": (
                "Projects are created to deliver business value. "
                "Four major triggers for project initiation are: regulatory or legal needs, "
                "stakeholder requests, technological advances, or improving and fixing "
                "existing processes and products."
            ),
            "question": "What is one common reason an organization starts a project?",
            "key_points": [
                "Regulation or legal compliance",
                "Stakeholder request",
                "New technology",
                "Process or product improvement",
            ],
            "min_points": 1,
            "next_unit": "lesson4_unit4_4",
        },
        {
            "id": "lesson4_unit4_4",
            "lesson": 4,
            "unit": "4.4",
            "title": "What is project management?",
            "material": (
                "Project management applies knowledge, skills, tools, and techniques "
                "to meet project requirements. The project manager gathers requirements, "
                "manages expectations, communicates frequently, coordinates stakeholders, "
                "and balances constraints such as scope, cost, schedule, quality, "
                "resources, and risk. Progressive elaboration means refining details "
                "as the project becomes clearer."
            ),
            "question": "What does progressive elaboration mean?",
            "key_points": [
                "Start broad and refine over time",
                "Details become clearer as the project progresses",
            ],
            "min_points": 1,
            "next_unit": "lesson4_unit4_5",
        },
        {
            "id": "lesson4_unit4_5",
            "lesson": 4,
            "unit": "4.5",
            "title": "Application areas",
            "material": (
                "Project management principles apply across industries: IT, healthcare, "
                "construction, government, manufacturing, and more. "
                "Regardless of industry, the core process groups remain the same: "
                "Initiating, Planning, Executing, Monitoring and Controlling, and Closing."
            ),
            "question": "Do all industries use the same project management principles?",
            "key_points": [
                "Yes, project management principles are universal",
                "Application can differ but core concepts stay the same",
            ],
            "min_points": 1,
            "next_unit": "lesson4_unit4_6",
        },
        {
            "id": "lesson4_unit4_6",
            "lesson": 4,
            "unit": "4.6",
            "title": "Project life cycle vs project management life cycle",
            "material": (
                "The Project Management Life Cycle is always the same: Initiate, Plan, "
                "Execute, Monitor and Control, and Close. "
                "The Project Life Cycle depends on the type of work and contains unique phases, "
                "such as foundation or framing in construction."
            ),
            "question": "How is the Project Management Life Cycle different from the Project Life Cycle?",
            "key_points": [
                "The Project Management Life Cycle is a fixed set of process groups",
                "The Project Life Cycle has unique phases depending on the type of work",
            ],
            "min_points": 1,
            "next_unit": None,  # 最後一題
        },
    ],
}

LESSON_5 = {
    "id": "lesson5",
    "title": "Related Areas of Project Management",
    "units": [
        {
            "id": "lesson5_unit5_1",
            "lesson": 5,
            "unit": "5.1",
            "title": "Program management",
            "material": (
                "A program is a group of related projects managed together to achieve benefits "
                "that individual projects cannot produce alone. Program managers coordinate "
                "schedules, resources, logistics, and cross-project communication."
            ),
            "question": "What is a program in project management?",
            "key_points": [
                "A group of related projects",
                "Managed together for shared benefits",
            ],
            "min_points": 1,
            "next_unit": "lesson5_unit5_2",
        },
        {
            "id": "lesson5_unit5_2",
            "lesson": 5,
            "unit": "5.2",
            "title": "Portfolio management",
            "material": (
                "A portfolio contains all organizational investments: projects, programs, "
                "and sometimes operations. Portfolio managers make decisions based on "
                "ROI, strategic fit, success factors, and risks."
            ),
            "question": "How is a portfolio different from a program?",
            "key_points": [
                "A portfolio represents all investments in projects, programs, and possibly operations",
                "A program is a group of related projects only",
            ],
            "min_points": 1,
            "next_unit": "lesson5_unit5_3",
        },
        {
            "id": "lesson5_unit5_3",
            "lesson": 5,
            "unit": "5.3",
            "title": "Project Management Office (PMO)",
            "material": (
                "A Project Management Office (PMO) provides governance, templates, shared resources, "
                "training, and project audits. There are three main PMO types: "
                "Supportive (guidance), Controlling (enforces processes), and Directive "
                "(directly manages projects)."
            ),
            "question": "What is one function of a PMO?",
            "key_points": [
                "Providing governance or standards",
                "Offering templates, tools, or methodologies",
                "Training and mentoring project managers",
                "Coordinating shared resources",
                "Performing project audits",
            ],
            "min_points": 1,
            "next_unit": "lesson5_unit5_4",
        },
        {
            "id": "lesson5_unit5_4",
            "lesson": 5,
            "unit": "5.4",
            "title": "Projects vs operations",
            "material": (
                "Projects are temporary and drive change. Operations are ongoing and repetitive, "
                "supporting daily business activities such as procurement, finance, and customer service. "
                "Both require people and resources but serve different purposes."
            ),
            "question": "What is one key difference between projects and operations?",
            "key_points": [
                "Projects are temporary",
                "Operations are ongoing",
            ],
            "min_points": 1,
            "next_unit": "lesson5_unit5_5",
        },
        {
            "id": "lesson5_unit5_5",
            "lesson": 5,
            "unit": "5.5",
            "title": "Organizational Project Management (OPM)",
            "material": (
                "Organizational Project Management (OPM) ensures all projects and programs "
                "align with strategy. It covers goals and tactics, value-based decisions, "
                "result delivery, and business value realization. "
                "OPM creates consistency and improves outcomes."
            ),
            "question": "Why do organizations use OPM?",
            "key_points": [
                "To align projects and programs with strategy",
                "To create consistency in how projects are managed",
                "To improve performance and business value",
            ],
            "min_points": 1,
            "next_unit": "lesson5_unit5_6",
        },
        {
            "id": "lesson5_unit5_6",
            "lesson": 5,
            "unit": "5.6",
            "title": "Project environments",
            "material": (
                "Project environments include physical, social, cultural, and organizational elements. "
                "Physical factors include location, weather, safety rules, and infrastructure. "
                "Social and cultural influences include values, ethics, politics, and organizational culture. "
                "Organizational factors include governance, policies, communication rules, structure, "
                "and resource availability."
            ),
            "question": "What is one example of a physical project environment factor?",
            "key_points": [
                "Location",
                "Weather",
                "Safety requirements",
                "Access limitations",
                "Facilities or equipment",
            ],
            "min_points": 1,
            "next_unit": "lesson5_unit5_7",
        },
        {
            "id": "lesson5_unit5_7",
            "lesson": 5,
            "unit": "5.7",
            "title": "Know your terms",
            "material": (
                "Knowing terminology is essential for PMP success. "
                "Questions often use synonyms or distractors. "
                "Understanding key terms improves clarity, decision-making, and accuracy."
            ),
            "question": "Why is knowing terminology important for the PMP exam?",
            "key_points": [
                "It helps you understand exam questions",
                "It avoids confusion with similar terms",
                "It improves accuracy and decision-making",
            ],
            "min_points": 1,
            "next_unit": None,
        },
    ],
}

LESSONS = {
    "lesson4": LESSON_4,
    "lesson5": LESSON_5,
}
