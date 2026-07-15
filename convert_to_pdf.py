from fpdf import FPDF
import os

# 配置
TEXT_FILENAME = "中华人民共和国个人信息保护法样例.txt"
FONT_PATH = "fonts/font.ttf"  # 请确保此处字体文件存在
OUTPUT_PDF = TEXT_FILENAME.replace(".txt", ".pdf")

class PDF(FPDF):
    def header(self):
        self.set_font("zh", size=12)
        self.cell(0, 10, "个人信息保护法样例", align="C", new_x="LMARGIN", new_y="NEXT")

    def footer(self):
        self.set_y(-15)
        self.set_font("zh", size=8)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")

def txt_to_pdf(txt_path, pdf_path, font_path):
    # 检查字体
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"找不到字体文件: {font_path}，请放置中文字体。")
    
    # 读取文本
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    pdf = PDF()
    pdf.add_font("zh", "", font_path, uni=True)  # 注册中文字体
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("zh", size=10)
    
    # 逐行写入，处理换行
    for line in text.splitlines():
        if not line.strip():
            pdf.ln(5)  # 空行
        else:
            pdf.multi_cell(0, 6, line, align="L")
    
    pdf.output(pdf_path)
    print(f"✅ PDF 已生成: {pdf_path}")

if __name__ == "__main__":
    txt_to_pdf(TEXT_FILENAME, OUTPUT_PDF, FONT_PATH)