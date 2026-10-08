/**
 * Kiểm và chụp giao diện Gradio đang chạy thật trên localhost.
 * Không mở app, train, tải model hay tạo kết quả thay cho mô hình.
 *
 * Ví dụ: node tools/capture_demo.js --url http://127.0.0.1:7860 --output reports/demo_ui
 * Node và Playwright có sẵn trong Codex; --playwright-path dùng cho máy khác.
 */
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const crypto = require("node:crypto");

const ROOT = path.resolve(__dirname, "..");
const SAMPLE = "Thank you so much! I am really happy with your help.";
const HEADERS = ["Cảm xúc", "Điểm", "Ngưỡng", "Được chọn"];
const TABLE_LABEL = "Điểm của 28 nhãn";
const RESULT_PATTERN = /Cảm xúc dự đoán:|Chưa có nhãn nào đạt ngưỡng\./;

function parseArguments(argv) {
  const options = {
    url: "http://127.0.0.1:7860",
    output: path.join(ROOT, "reports", "demo_ui"),
    playwrightPath: process.env.CODEX_PLAYWRIGHT_PATH || path.join(
      os.homedir(), ".cache", "codex-runtimes", "codex-primary-runtime",
      "dependencies", "node", "node_modules", "playwright"
    ),
    browserPath: null,
  };
  const names = {
    "--url": "url",
    "--output": "output",
    "--playwright-path": "playwrightPath",
    "--browser-path": "browserPath",
  };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--help" || argv[i] === "-h") {
      return null;
    }
    if (!names[argv[i]] || !argv[i + 1] || argv[i + 1].startsWith("--")) {
      throw new Error(`Tham số không hợp lệ hoặc thiếu giá trị: ${argv[i]}`);
    }
    options[names[argv[i]]] = argv[++i];
  }
  const address = new URL(options.url);
  if (!["http:", "https:"].includes(address.protocol) ||
      !["localhost", "127.0.0.1", "[::1]"].includes(address.hostname)) {
    throw new Error("--url phải trỏ tới app localhost/127.0.0.1/::1 đã mở.");
  }
  options.output = path.resolve(options.output);
  options.playwrightPath = path.resolve(options.playwrightPath);
  if (options.browserPath) options.browserPath = path.resolve(options.browserPath);
  return options;
}

function chooseBrowserPath(explicitPath) {
  if (explicitPath) {
    if (!fs.existsSync(explicitPath)) throw new Error(`Không có browser: ${explicitPath}`);
    return explicitPath;
  }
  // Cache đã có trên máy làm đồ án. Máy khác có thể dùng cache mặc định Playwright.
  const cached = path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"),
    "ms-playwright", "chromium_headless_shell-1243", "chrome-headless-shell-win64",
    "chrome-headless-shell.exe");
  return fs.existsSync(cached) ? cached : null;
}

function readLabels() {
  const labels = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "labels.json"), "utf8"));
  if (!Array.isArray(labels) || labels.length !== 28 || new Set(labels).size !== 28) {
    throw new Error("data/labels.json phải chứa đúng 28 nhãn khác nhau.");
  }
  return labels;
}

async function capture(options) {
  const labels = readLabels();
  let playwright;
  try {
    playwright = require(options.playwrightPath);
  } catch (error) {
    throw new Error(`Không đọc được Playwright tại ${options.playwrightPath}. ` +
      `Truyền --playwright-path thư mục đã cài sẵn. Chi tiết: ${error.message}`);
  }
  const browserPath = chooseBrowserPath(options.browserPath);
  const browser = await playwright.chromium.launch({
    headless: true,
    ...(browserPath ? { executablePath: browserPath } : {}),
  });
  const errors = [];
  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1100 },
      deviceScaleFactor: 1,
      locale: "vi-VN",
    });
    const page = await context.newPage();
    page.setDefaultTimeout(120000);
    page.on("pageerror", error => errors.push(error.message));
    const response = await page.goto(options.url, { waitUntil: "domcontentloaded" });
    if (!response || !response.ok()) {
      throw new Error(`App chưa truy cập được: HTTP ${response ? response.status() : "không có phản hồi"}`);
    }

    const input = page.getByRole("textbox", { name: "Bình luận tiếng Anh", exact: true });
    const button = page.getByRole("button", { name: "Nhận diện cảm xúc", exact: true });
    await input.fill(SAMPLE);
    await button.click();

    // Markdown rỗng lúc mới mở app. Chỉ nhận kết quả được trả sau lần bấm thật.
    const result = page.locator(".prose").filter({ hasText: RESULT_PATTERN }).last();
    await result.waitFor({ state: "visible" });
    const resultText = (await result.innerText()).trim();
    if (!RESULT_PATTERN.test(resultText)) throw new Error("Chưa có thông báo dự đoán thật.");

    // .table-container là wrapper trong frontend Gradio hiện tại. Bảng aria-hidden
    // dùng riêng để đo kích thước; chỉ đọc bảng virtual thật, tránh hàng/headers lặp.
    const table = page.locator(".table-container").filter({ hasText: TABLE_LABEL }).first();
    await table.waitFor({ state: "visible" });
    for (const header of HEADERS) {
      await table.locator("table:not([aria-hidden='true']) thead")
        .getByText(header, { exact: true }).first().waitFor({ state: "visible" });
    }

    // Bảng có thể cuộn/virtualize. Kiểm các hàng thật có trong DOM và ghi số đã kiểm;
    // không gọi việc nhìn tiêu đề là kiểm số học của cả 28 nhãn.
    const readRows = async () => table.evaluate((element, expectedLabels) => {
      const allowed = new Set(expectedLabels);
      const rows = Array.from(element.querySelectorAll("table:not([aria-hidden='true']) tbody tr"));
      return rows.map(row => {
        const cells = Array.from(row.querySelectorAll("td, th, [role='gridcell']"))
          .map(cell => (cell.innerText || cell.textContent || "").trim());
        const labelIndex = cells.findIndex(cell => allowed.has(cell));
        if (labelIndex < 0 || cells.length < labelIndex + 3) return null;
        return { label: cells[labelIndex], score: cells[labelIndex + 1] === "" ? null : Number(cells[labelIndex + 1]),
          threshold: cells[labelIndex + 2] === "" ? null : Number(cells[labelIndex + 2]), selected: cells[labelIndex + 3] || "" };
      }).filter(Boolean);
    }, labels);
    // Hai output Markdown/bảng có thể render ở hai thời điểm; chờ bảng có hàng thật.
    await page.waitForFunction(({ expectedLabels, tableLabel }) => {
      const elements = Array.from(document.querySelectorAll(".table-container"));
      const element = elements.find(item => item.textContent.includes(tableLabel));
      if (!element) return false;
      return Array.from(element.querySelectorAll("table:not([aria-hidden='true']) td"))
        .some(cell => expectedLabels.includes((cell.innerText || cell.textContent || "").trim()));
    }, { expectedLabels: labels, tableLabel: TABLE_LABEL });
    const renderedRows = await readRows();
    if (!renderedRows.length || new Set(renderedRows.map(row => row.label)).size !== renderedRows.length) {
      throw new Error("Bảng chưa có hàng cảm xúc thật, hoặc có nhãn lặp. Kiểm DOM Gradio hiện tại.");
    }
    for (const row of renderedRows) {
      if (![row.score, row.threshold].every(value => Number.isFinite(value) && value >= 0 && value <= 1)) {
        throw new Error(`Điểm/ngưỡng hiển thị không hợp lệ: ${row.label}`);
      }
    }
    if (errors.length) throw new Error(`Lỗi frontend: ${errors.join("; ")}`);

    await page.evaluate(async () => { if (document.fonts) await document.fonts.ready; });
    const modelText = (await page.locator(".prose").filter({ hasText: "Mô hình:" }).first().innerText()).trim();
    const title = await page.title();
    const checkedAt = new Date().toISOString();
    // Chỉ tạo ảnh/evidence PASS sau toàn bộ các kiểm UI phía trên thành công.
    fs.mkdirSync(options.output, { recursive: true });
    const screenshotPath = path.join(options.output, "demo_ui.png");
    await page.screenshot({ path: screenshotPath, fullPage: true, animations: "disabled" });
    const screenshotHash = crypto.createHash("sha256").update(fs.readFileSync(screenshotPath)).digest("hex");
    const evidence = {
      schema_version: 1,
      checked_at_utc: checkedAt,
      interface_status: "PASS",
      url: page.url(),
      http_status: response.status(),
      page_title: title,
      sample_text: SAMPLE,
      button_clicked: "Nhận diện cảm xúc",
      model_display: modelText,
      prediction_display: resultText,
      table_label: TABLE_LABEL,
      headers_checked: HEADERS,
      advertised_label_count: 28,
      rendered_rows_checked: renderedRows,
      rendered_row_count: renderedRows.length,
      all_28_rows_checked: renderedRows.length === 28 && labels.every(label => renderedRows.some(row => row.label === label)),
      model_score_equivalence: "NOT_CHECKED_HERE",
      screenshot: "demo_ui.png",
      screenshot_sha256: screenshotHash,
      browser_version: browser.version(),
      playwright_path: options.playwrightPath,
      browser_path: browserPath,
      frontend_errors: errors,
      note: "Kiểm tương tác và bảng trên app thật. Không thay thế verify_demo.py, test metrics hoặc kiểm checkpoint/ngưỡng.",
    };
    const evidencePath = path.join(options.output, "evidence.json");
    const temporaryPath = evidencePath + ".tmp";
    fs.writeFileSync(temporaryPath, JSON.stringify(evidence, null, 2) + "\n", "utf8");
    fs.renameSync(temporaryPath, evidencePath);
    console.log(`UI PASS: ${evidencePath}\nẢnh thật: ${screenshotPath}\nHàng đã kiểm trong DOM: ${renderedRows.length}/28.`);
  } finally {
    await browser.close();
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (!options) {
    console.log("node tools/capture_demo.js [--url http://127.0.0.1:7860] " +
      "[--output reports/demo_ui] [--playwright-path path/to/playwright] [--browser-path path/to/chromium]");
    return;
  }
  await capture(options);
}

main().catch(error => {
  console.error(`UI chưa PASS; không ghi evidence mới: ${error.message}`);
  process.exitCode = 1;
});
