"""Generate USER_SETUP_INSTRUCTIONS.pdf — step-by-step guide for Balaji."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parents[2] / "USER_SETUP_INSTRUCTIONS.pdf"

styles = getSampleStyleSheet()

H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], textColor=colors.HexColor("#1f3a5f"),
    fontSize=20, spaceAfter=10, leading=24,
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], textColor=colors.HexColor("#1f3a5f"),
    fontSize=15, spaceAfter=8, spaceBefore=14, leading=18,
)
H3 = ParagraphStyle(
    "H3", parent=styles["Heading3"], textColor=colors.HexColor("#2b5d8a"),
    fontSize=12, spaceAfter=4, spaceBefore=8, leading=15,
)
BODY = ParagraphStyle(
    "Body", parent=styles["BodyText"], fontSize=10.5, leading=14,
    spaceAfter=6, alignment=TA_LEFT,
)
NOTE = ParagraphStyle(
    "Note", parent=BODY, backColor=colors.HexColor("#fff8e1"),
    borderColor=colors.HexColor("#f0c674"), borderWidth=1, borderPadding=8,
    leftIndent=4, rightIndent=4, spaceBefore=6, spaceAfter=10,
)
CODE = ParagraphStyle(
    "Code", parent=styles["Code"], fontName="Courier", fontSize=9,
    leading=11, backColor=colors.HexColor("#f4f4f4"), borderPadding=6,
    leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=8,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY, leftIndent=18, bulletIndent=6, spaceAfter=3,
)


def code(text: str) -> Paragraph:
    safe = (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\n", "<br/>")
    )
    return Paragraph(safe, CODE)


def bullet(text: str) -> Paragraph:
    return Paragraph("• " + text, BULLET)


def note(text: str) -> Paragraph:
    return Paragraph("<b>Note:</b> " + text, NOTE)


def hr() -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"),
                      spaceBefore=8, spaceAfter=8)


def make_table(header, rows, widths=None):
    data = [header] + rows
    t = Table(data, colWidths=widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
            ]
        )
    )
    return t


def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="Fraud Detection Pipeline — User Setup Instructions",
        author="Generated for Balaji Viswanathan",
    )
    s = []

    # ---------- Cover ----------
    s.append(Paragraph("Fraud Detection Pipeline", H1))
    s.append(Paragraph("User Setup Instructions — what YOU need to do", H2))
    s.append(hr())
    s.append(Paragraph(
        "Claude has scaffolded the entire project at "
        "<font name='Courier'>fraud-detection-pipeline/</font>. This document walks you through the "
        "manual steps that require your accounts, credentials, or local machine — things Claude cannot do "
        "for you. Follow the sections in order.", BODY))
    s.append(Spacer(1, 6))

    s.append(Paragraph("What's already done (by Claude)", H3))
    for b in [
        "Complete project structure — 5 Docker services (training, serving, monitoring, UI, MLflow)",
        "All Python source code: data loading, feature engineering, training, FastAPI, Streamlit, Evidently drift",
        "Docker Compose configuration that orchestrates everything",
        "Tests, CI workflow, EC2 bootstrap script, CloudWatch agent config",
        "Project log file (PROJECT_LOG.md) — append entries as you progress",
        "README, .gitignore, Makefile",
    ]:
        s.append(bullet(b))

    s.append(Paragraph("What you need to do (overview)", H3))
    overview = [
        ["#", "Step", "Why it needs you"],
        ["1", "Install Docker Desktop on Windows", "Local install, requires your machine"],
        ["2", "Create a Kaggle account and download the dataset", "Auth-protected, ~720 MB"],
        ["3", "Run the project locally and trigger training", "Validates everything before AWS"],
        ["4", "Create a GitHub repo and push the code", "Your account + credentials"],
        ["5", "Set up AWS account + IAM user", "Your AWS billing"],
        ["6", "Launch EC2, deploy, configure CloudWatch", "Your AWS resources"],
        ["7", "Polish portfolio: README, screenshots, demo video", "Your judgment"],
    ]
    s.append(make_table(overview[0], overview[1:], widths=[0.4 * inch, 2.6 * inch, 4 * inch]))
    s.append(PageBreak())

    # ---------- Step 1 ----------
    s.append(Paragraph("Step 1 — Install Docker Desktop (Windows)", H2))
    s.append(Paragraph("Docker is required to build and run the 5 containers.", BODY))
    for b in [
        "Go to <font name='Courier'>https://www.docker.com/products/docker-desktop/</font> and download Docker Desktop for Windows.",
        "Run the installer. When prompted, enable WSL 2 backend (recommended on Windows 11).",
        "Reboot if prompted.",
        "Open Docker Desktop, wait for the whale icon to say 'Engine running'.",
        "Open PowerShell or Git Bash and verify:",
    ]:
        s.append(bullet(b))
    s.append(code("docker --version\ndocker compose version"))
    s.append(note("If <b>docker compose</b> is not found, you have an older Docker. Update Docker Desktop "
                  "to a recent version (the Compose plugin is bundled)."))

    # ---------- Step 2 ----------
    s.append(Paragraph("Step 2 — Download the IEEE-CIS Fraud Detection dataset", H2))
    s.append(Paragraph("This dataset is hosted on Kaggle and requires a free account + competition rules acceptance.", BODY))
    s.append(Paragraph("Option A — Browser download (simplest)", H3))
    for b in [
        "Sign up / log in at <font name='Courier'>https://www.kaggle.com/</font>",
        "Open <font name='Courier'>https://www.kaggle.com/c/ieee-fraud-detection/data</font>",
        "Click <b>Late Submission</b> or <b>Join</b> to accept the competition rules (required to access the data).",
        "Click <b>Download All</b>. You'll get a ~120 MB zip; unzip it.",
        "Copy <font name='Courier'>train_transaction.csv</font> and <font name='Courier'>train_identity.csv</font> "
        "into <font name='Courier'>fraud-detection-pipeline/data/raw/</font>",
    ]:
        s.append(bullet(b))

    s.append(Paragraph("Option B — Kaggle CLI (faster on EC2)", H3))
    s.append(code(
        "pip install kaggle\n"
        "# Get API token from https://www.kaggle.com/settings (Create New Token)\n"
        "# Save the downloaded kaggle.json to:\n"
        "#   Windows: %USERPROFILE%\\.kaggle\\kaggle.json\n"
        "#   Linux:   ~/.kaggle/kaggle.json  (chmod 600)\n"
        "kaggle competitions download -c ieee-fraud-detection -p data/raw/\n"
        "cd data/raw && unzip ieee-fraud-detection.zip"
    ))
    s.append(note("You only need <b>train_transaction.csv</b> and <b>train_identity.csv</b>. "
                  "The test_*.csv files are not needed for this project — we make our own time-based test split."))

    s.append(PageBreak())

    # ---------- Step 3 ----------
    s.append(Paragraph("Step 3 — Run the project locally and train", H2))
    s.append(Paragraph(
        "Validate everything works on your laptop before paying for AWS. Open a terminal in the project root.", BODY))

    s.append(Paragraph("3.1 — First-time setup", H3))
    s.append(code(
        'cd "C:/All Files/Projects/AI_Projects/New folder/fraud-detection-pipeline"\n'
        "cp .env.example .env\n"
        "# (optional) edit .env if you want to change THRESHOLD or AWS_DEFAULT_REGION"
    ))

    s.append(Paragraph("3.2 — Build and start all services", H3))
    s.append(code("docker compose up -d --build"))
    s.append(Paragraph("First build takes 5–10 minutes (downloads Python images, installs xgboost / lightgbm / shap / evidently). After that, restarts are fast.", BODY))

    s.append(Paragraph("3.3 — Verify all services are up", H3))
    s.append(code(
        "docker compose ps\n"
        "# All services should show 'running'.\n"
        "# Open in your browser:\n"
        "#   http://localhost:8501  (Streamlit dashboard)\n"
        "#   http://localhost:5000  (MLflow)\n"
        "#   http://localhost:8000/docs  (FastAPI Swagger — will be 503 until model trained)"
    ))

    s.append(Paragraph("3.4 — Trigger training (one-time, ~30–60 minutes on a laptop)", H3))
    s.append(code(
        "curl -X POST http://localhost:8001/train\n"
        "# Poll status:\n"
        "curl http://localhost:8001/status\n"
        "# Or watch logs live:\n"
        "docker compose logs -f training"
    ))
    s.append(note("If training is too slow on your laptop, set <font name='Courier'>SAMPLE_FRAC=0.1</font> "
                  "in .env to train on 10%% of the data first, then run the full version on EC2."))

    s.append(Paragraph("3.5 — Once training finishes", H3))
    for b in [
        "Open <b>http://localhost:8501</b> → Predictions page → submit a transaction.",
        "Open <b>http://localhost:5000</b> to see the MLflow runs (one per model).",
        "Open <b>http://localhost:8501</b> → Drift Monitoring → click 'Run drift check now'.",
        "Append a log entry to <b>PROJECT_LOG.md</b> noting metrics + best model.",
    ]:
        s.append(bullet(b))

    s.append(Paragraph("3.6 — Stop / clean up", H3))
    s.append(code(
        "docker compose down       # stop containers (keeps models)\n"
        "docker compose down -v    # also remove volumes (deletes trained models)"
    ))

    s.append(PageBreak())

    # ---------- Step 4 ----------
    s.append(Paragraph("Step 4 — Push to GitHub", H2))
    for b in [
        "Create a new repo at <font name='Courier'>https://github.com/new</font> — name it "
        "<font name='Courier'>fraud-detection-pipeline</font>, public.",
        "Do NOT initialize with README/license (we already have them).",
        "From the project root run the commands below — replace YOUR_USERNAME.",
    ]:
        s.append(bullet(b))
    s.append(code(
        'cd "C:/All Files/Projects/AI_Projects/New folder/fraud-detection-pipeline"\n'
        "git init\n"
        "git add .\n"
        'git commit -m "Initial scaffold for fraud detection pipeline"\n'
        "git branch -M main\n"
        "git remote add origin https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git\n"
        "git push -u origin main"
    ))
    s.append(note("Your dataset CSVs and .env are already in <b>.gitignore</b>. Do <i>not</i> commit them. "
                  "Confirm with <font name='Courier'>git status</font> before push — only source files should appear."))

    # ---------- Step 5 ----------
    s.append(Paragraph("Step 5 — AWS account setup", H2))
    s.append(Paragraph("If you already used AWS for the Compliance Auditor project, skip to Step 6.", BODY))
    for b in [
        "Sign up at <font name='Courier'>https://aws.amazon.com/</font> (need credit card; free tier covers most of this).",
        "Enable MFA on the root account (IAM → Security credentials).",
        "Create an IAM user named <font name='Courier'>fraud-detection-deployer</font> with "
        "AmazonEC2FullAccess + CloudWatchFullAccess + IAMReadOnlyAccess. Save the access keys.",
        "Region: pick <b>us-east-2 (Ohio)</b> to match your existing instance.",
    ]:
        s.append(bullet(b))

    s.append(PageBreak())

    # ---------- Step 6 ----------
    s.append(Paragraph("Step 6 — Deploy to EC2", H2))

    s.append(Paragraph("6.1 — Create IAM role for the EC2 instance", H3))
    for b in [
        "AWS Console → IAM → Roles → Create role.",
        "Trusted entity: AWS service → EC2.",
        "Attach policies: <b>CloudWatchAgentServerPolicy</b> and <b>CloudWatchFullAccess</b>.",
        "Name: <font name='Courier'>fraud-detection-ec2-role</font>. Create.",
    ]:
        s.append(bullet(b))

    s.append(Paragraph("6.2 — Launch the EC2 instance", H3))
    specs = [
        ["Setting", "Value"],
        ["Name", "fraud-detection"],
        ["AMI", "Amazon Linux 2023 (free tier eligible)"],
        ["Instance type", "t3.medium  (2 vCPU, 4 GB RAM)"],
        ["Key pair", "Reuse existing OR create fraud-detection-key.pem"],
        ["Storage", "50 GB gp3 (NOT 30 GB — dataset is large)"],
        ["IAM instance profile", "fraud-detection-ec2-role (from 6.1)"],
        ["Region", "us-east-2 (Ohio)"],
    ]
    s.append(make_table(specs[0], specs[1:], widths=[2 * inch, 5 * inch]))

    s.append(Paragraph("6.3 — Security group", H3))
    s.append(Paragraph("Create or reuse a security group with these inbound rules. Set <b>Source: My IP</b> "
                       "for everything except the SSH rule (also My IP):", BODY))
    sg = [
        ["Port", "Protocol", "Purpose"],
        ["22", "TCP", "SSH"],
        ["8000", "TCP", "Serving FastAPI"],
        ["8001", "TCP", "Training service (admin)"],
        ["8002", "TCP", "Monitoring service"],
        ["8501", "TCP", "Streamlit UI"],
        ["5000", "TCP", "MLflow UI"],
    ]
    s.append(make_table(sg[0], sg[1:], widths=[1 * inch, 1.5 * inch, 4.5 * inch]))

    s.append(Paragraph("6.4 — Allocate Elastic IP", H3))
    s.append(Paragraph("EC2 Console → Elastic IPs → Allocate → Associate with the new instance. This gives a stable IP across reboots.", BODY))

    s.append(Paragraph("6.5 — SSH in and bootstrap", H3))
    s.append(code(
        "ssh -i fraud-detection-key.pem ec2-user@YOUR_ELASTIC_IP\n"
        "# Once in:\n"
        "git clone https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git\n"
        "cd fraud-detection-pipeline\n"
        "bash scripts/setup_ec2.sh\n"
        "# Logout and SSH in again (so docker group takes effect)\n"
        "exit"
    ))

    s.append(Paragraph("6.6 — Upload the dataset", H3))
    s.append(Paragraph("Either <b>scp</b> from your laptop OR install Kaggle CLI on the instance "
                       "(see Step 2 Option B).", BODY))
    s.append(code(
        "# From your laptop (Git Bash on Windows):\n"
        'scp -i fraud-detection-key.pem "C:/.../data/raw/train_transaction.csv" \\\n'
        "  ec2-user@YOUR_ELASTIC_IP:~/fraud-detection-pipeline/data/raw/\n"
        'scp -i fraud-detection-key.pem "C:/.../data/raw/train_identity.csv" \\\n'
        "  ec2-user@YOUR_ELASTIC_IP:~/fraud-detection-pipeline/data/raw/"
    ))

    s.append(Paragraph("6.7 — Build and start", H3))
    s.append(code(
        "# SSH back in:\n"
        "ssh -i fraud-detection-key.pem ec2-user@YOUR_ELASTIC_IP\n"
        "cd fraud-detection-pipeline\n"
        "cp .env.example .env\n"
        "docker compose up -d --build       # 10-15 minutes first time\n"
        "docker compose ps\n"
        "curl -X POST http://localhost:8001/train     # kicks off training\n"
        "docker compose logs -f training              # watch progress"
    ))

    s.append(Paragraph("6.8 — Open in your browser", H3))
    s.append(code(
        "http://YOUR_ELASTIC_IP:8501   # Streamlit dashboard\n"
        "http://YOUR_ELASTIC_IP:5000   # MLflow\n"
        "http://YOUR_ELASTIC_IP:8000/docs  # FastAPI Swagger"
    ))

    s.append(PageBreak())

    # ---------- Step 7 ----------
    s.append(Paragraph("Step 7 — CloudWatch dashboard + alarms", H2))
    s.append(Paragraph("The monitoring service already publishes <b>DriftScore</b> and <b>FeatureDriftCount</b> "
                       "to CloudWatch under namespace <font name='Courier'>FraudDetection</font>. You set up "
                       "the dashboard + alarms manually in the console.", BODY))

    s.append(Paragraph("7.1 — Install CloudWatch Agent (for system metrics + container logs)", H3))
    s.append(code(
        "# On the EC2 instance:\n"
        "sudo dnf install -y amazon-cloudwatch-agent\n"
        "sudo cp ~/fraud-detection-pipeline/cloudwatch/cloudwatch-config.json \\\n"
        "  /opt/aws/amazon-cloudwatch-agent/etc/cloudwatch-config.json\n"
        "sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \\\n"
        "  -a fetch-config -m ec2 -s -c file:/opt/aws/amazon-cloudwatch-agent/etc/cloudwatch-config.json"
    ))

    s.append(Paragraph("7.2 — Create a dashboard", H3))
    for b in [
        "CloudWatch Console → Dashboards → Create dashboard.",
        "Add widgets: line chart for FraudDetection/DriftScore, FeatureDriftCount, EC2 CPUUtilization, mem_used_percent.",
    ]:
        s.append(bullet(b))

    s.append(Paragraph("7.3 — Create alarms", H3))
    alarms = [
        ["Alarm", "Condition", "Meaning"],
        ["High Drift", "DriftScore > 0.30 for 5 min", "Input distribution shifted significantly"],
        ["No Predictions", "Sum(fraud_predictions_total) = 0 for 15 min", "Serving down or no traffic"],
        ["High CPU", "CPUUtilization > 85%% for 10 min", "Instance under-provisioned"],
    ]
    s.append(make_table(alarms[0], alarms[1:], widths=[1.4 * inch, 2.6 * inch, 3 * inch]))
    s.append(Paragraph("Optionally hook each alarm to an SNS topic that emails you.", BODY))

    # ---------- Step 8 ----------
    s.append(Paragraph("Step 8 — Polish for portfolio", H2))
    for b in [
        "Update README with your real metrics (AUC-PR, AUC-ROC) once training completes.",
        "Take screenshots: Streamlit prediction page, MLflow comparison, drift report.",
        "Record a 2–3 minute demo video walking through prediction + SHAP + drift.",
        "Add resume bullets (template is in section 20 of the project plan PDF).",
        "Post on LinkedIn with the architecture diagram.",
    ]:
        s.append(bullet(b))

    s.append(Paragraph("Stop the EC2 instance when not actively demoing — saves cost.", NOTE))

    # ---------- Logging + troubleshooting ----------
    s.append(Paragraph("How to use the project log", H2))
    s.append(Paragraph(
        "<font name='Courier'>fraud-detection-pipeline/PROJECT_LOG.md</font> is the running changelog. Append an "
        "entry every time you (or Claude) make a meaningful change — training run, deployment, bug, drift alert, "
        "decision. The file already has the initial scaffold entry. Append using this format:", BODY))
    s.append(code(
        "## 2026-05-04 — Trained baseline XGBoost\n"
        "- Dataset: full 590K rows\n"
        "- Best model: XGBoost, scale_pos_weight=20\n"
        "- Test AUC-PR=0.71, AUC-ROC=0.93, threshold=0.34\n"
        "- MLflow run id: abc123\n"
        "- Issue: SMOTE caused OOM on t3.medium — disabled, used class weights only"
    ))

    s.append(Paragraph("Common issues + fixes", H2))
    issues = [
        ["Symptom", "Fix"],
        ["docker compose: command not found",
         "Update Docker Desktop. Older versions need 'docker-compose' (hyphen)."],
        ["Training service errors with FileNotFoundError",
         "CSVs not in data/raw/. Place train_transaction.csv + train_identity.csv there."],
        ["Out of memory during training",
         "Set SAMPLE_FRAC=0.2 in .env; restart containers."],
        ["Serving returns 503",
         "Model not trained yet. POST /train on training-service first, wait for completion."],
        ["Streamlit can't reach serving",
         "Both must be on same docker network. 'docker compose up' handles this — don't run individually."],
        ["EC2 disk full",
         "Use 50 GB EBS, not 30 GB. Or 'docker system prune -a' to clean unused images."],
        ["Drift report empty",
         "Need predictions logged first. Submit a few transactions via UI, then re-run drift check."],
    ]
    s.append(make_table(issues[0], issues[1:], widths=[2.3 * inch, 4.7 * inch]))

    s.append(Spacer(1, 14))
    s.append(hr())
    s.append(Paragraph(
        "<i>Generated for Balaji Viswanathan — Real-Time Fraud Detection Pipeline.</i>", BODY))

    doc.build(s)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
